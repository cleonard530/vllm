// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

// Local NEON stand-in for the ATen Vectorized<> API used by cpu_types_arm.hpp.
// Header-only: no ATen/cpu/vec includes.

#include <cmath>
#include <cstring>
#include <tuple>

#include <arm_neon.h>

#ifdef ARM_BF16_SUPPORT
  #include <arm_bf16.h>
#endif

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

namespace vec_op::arm_vec {

#ifdef ARM_BF16_SUPPORT
using neon_bfloat16x8_t = bfloat16x8_t;
using neon_bfloat16x4_t = bfloat16x4_t;
#else
using neon_bfloat16x8_t = uint16x8_t;
using neon_bfloat16x4_t = uint16x4_t;
#endif

template <typename T>
struct Vectorized;

namespace {

template <typename T>
inline void copy_partial(T* dst, const T* src, int count) {
  std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(T));
}

inline float32x4_t bf16x4_to_f32(neon_bfloat16x4_t v) {
#ifdef ARM_BF16_SUPPORT
  return vcvt_f32_bf16(v);
#else
  return vreinterpretq_f32_u32(vshll_n_u16(v, 16));
#endif
}

inline neon_bfloat16x4_t f32_to_bf16x4(float32x4_t v) {
#ifdef ARM_BF16_SUPPORT
  return vcvt_bf16_f32(v);
#else
  return vshrn_n_u32(vreinterpretq_u32_f32(v), 16);
#endif
}

inline neon_bfloat16x8_t load_bf16x8(const c10::BFloat16* ptr) {
#ifdef ARM_BF16_SUPPORT
  return vld1q_bf16(reinterpret_cast<const bfloat16_t*>(ptr));
#else
  return vld1q_u16(reinterpret_cast<const uint16_t*>(ptr));
#endif
}

inline void store_bf16x8(c10::BFloat16* ptr, neon_bfloat16x8_t v) {
#ifdef ARM_BF16_SUPPORT
  vst1q_bf16(reinterpret_cast<bfloat16_t*>(ptr), v);
#else
  vst1q_u16(reinterpret_cast<uint16_t*>(ptr), v);
#endif
}

inline neon_bfloat16x4_t bf16_low(neon_bfloat16x8_t v) {
#ifdef ARM_BF16_SUPPORT
  return vget_low_bf16(v);
#else
  return vget_low_u16(v);
#endif
}

inline neon_bfloat16x4_t bf16_high(neon_bfloat16x8_t v) {
#ifdef ARM_BF16_SUPPORT
  return vget_high_bf16(v);
#else
  return vget_high_u16(v);
#endif
}

inline neon_bfloat16x8_t bf16_combine(neon_bfloat16x4_t lo,
                                      neon_bfloat16x4_t hi) {
#ifdef ARM_BF16_SUPPORT
  return vcombine_bf16(lo, hi);
#else
  return vcombine_u16(lo, hi);
#endif
}

inline neon_bfloat16x8_t bf16_dup(c10::BFloat16 v) {
  uint16_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
#ifdef ARM_BF16_SUPPORT
  // Workaround for GCC <= 13 miscompiling a constant bf16 broadcast: it can
  // lower a compile-time-constant bf16 dup to an FP16 immediate move (e.g.
  // writing 0x3C00 = ~0.0078 per lane) instead of materializing the actual
  // bf16 bit pattern (0x3F80 for 1.0). Fixed in gcc-14. A plain
  // memcpy/reinterpret isn't enough since gcc constant-tracks through it and
  // rematerializes the buggy immediate at the use site, so launder the bits
  // through an asm barrier first. See ATen's vec128_bfloat16_neon.h for the
  // same workaround.
  #if __GNUC__ <= 13 && !defined(__clang__)
  __asm__("" : "+r"(bits));
  #endif
  bfloat16_t h;
  std::memcpy(&h, &bits, sizeof(h));
  return vdupq_n_bf16(h);
#else
  return vdupq_n_u16(bits);
#endif
}

}  // namespace

template <>
struct Vectorized<float> {
  float32x4_t values;
  static constexpr int size_ = 4;

  Vectorized() : values(vdupq_n_f32(0.f)) {}
  Vectorized(float32x4_t v) : values(v) {}
  explicit Vectorized(float v) : values(vdupq_n_f32(v)) {}

  static constexpr int size() { return size_; }

  operator float32x4_t() const { return values; }

  static Vectorized loadu(const float* ptr) { return vld1q_f32(ptr); }
  static Vectorized loadu(const float* ptr, int count) {
    alignas(16) float buf[size_]{};
    copy_partial(buf, ptr, count);
    return loadu(buf);
  }

  void store(float* ptr) const { vst1q_f32(ptr, values); }
  void store(float* ptr, int count) const {
    alignas(16) float buf[size_];
    store(buf);
    copy_partial(ptr, buf, count);
  }

  Vectorized neg() const { return vnegq_f32(values); }
  Vectorized abs() const { return vabsq_f32(values); }
  Vectorized sqrt() const { return vsqrtq_f32(values); }
  Vectorized tanh() const { return vec_op::fast_tanhf_f32x4(values); }

  template <typename Fn>
  Vectorized map_scalar(Fn fn) const {
    alignas(16) float buf[size_];
    store(buf);
    for (int i = 0; i < size_; ++i) {
      buf[i] = fn(buf[i]);
    }
    return loadu(buf);
  }

  // Implementation adapted from Arm Optimized Routines
  // https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/advsimd/expf.c
  // (also used by ATen's NEON vec backend). Falls back to scalar std::exp
  // for the rare inputs where |x| > 87.3..., outside this polynomial's
  // valid range.
  Vectorized exp_u20() const {
    const float32x4_t special_bound = vdupq_n_f32(0x1.5d5e2ap+6f);
    uint32x4_t cmp = vcagtq_f32(values, special_bound);
    if (vpaddd_u64(vreinterpretq_u64_u32(cmp)) != 0) {
      return map_scalar([](float x) { return std::exp(x); });
    }

    const float32x4_t inv_ln2 = vdupq_n_f32(0x1.715476p+0f);
    constexpr float ln2_hi = 0x1.62e4p-1f;
    constexpr float ln2_lo = 0x1.7f7d1cp-20f;
    constexpr float c0 = 0x1.0e4020p-7f;
    constexpr float c2 = 0x1.555e66p-3f;
    const float32x4_t ln2_c02 = {ln2_hi, ln2_lo, c0, c2};

    const uint32x4_t exponent_bias = vdupq_n_u32(0x3f800000);
    const float32x4_t c1 = vdupq_n_f32(0x1.573e2ep-5f);
    const float32x4_t c3 = vdupq_n_f32(0x1.fffdb6p-2f);
    const float32x4_t c4 = vdupq_n_f32(0x1.ffffecp-1f);

    // exp(x) = 2^n (1 + poly(r)), with 1 + poly(r) in [1/sqrt(2), sqrt(2)]
    // x = ln2*n + r, with r in [-ln2/2, ln2/2].
    float32x4_t n = vrndaq_f32(vmulq_f32(values, inv_ln2));
    float32x4_t r = vfmsq_laneq_f32(values, n, ln2_c02, 0);
    r = vfmsq_laneq_f32(r, n, ln2_c02, 1);
    uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), 23);
    float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));

    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t p = vfmaq_laneq_f32(c1, r, ln2_c02, 2);
    float32x4_t q = vfmaq_laneq_f32(c3, r, ln2_c02, 3);
    q = vfmaq_f32(q, p, r2);
    p = vmulq_f32(c4, r);
    float32x4_t poly = vfmaq_f32(p, q, r2);

    return vfmaq_f32(scale, poly, scale);
  }

  // Implementation adapted from ATen's NEON vec backend: a fast, less
  // precise exp variant intended for cases where outputs will be downcast
  // to FP16/BF16 (e.g. attention softmax). Accurate within 1 ULP for
  // FP16/BF16 for inputs in [-87.683, 88.376]; clamps outside that range to
  // 0 / inf instead of over/underflowing.
  Vectorized fexp_u20() const {
    const float32x4_t lower_bound = vdupq_n_f32(-0x1.5ebb82p+6f);
    const float32x4_t upper_bound = vdupq_n_f32(0x1.61814ap+6f);
    const float32x4_t inv_ln2 = vdupq_n_f32(0x1.715476p+0f);
    constexpr float ln2 = 0x1.62e43p-1f;
    constexpr float c2 = 0x1.5592ecp-3f;
    const float32x4_t c3 = vdupq_n_f32(0x1.017d34p-1f);
    const uint32x4_t lt_lower = vcltq_f32(values, lower_bound);
    const uint32x4_t gt_upper = vcgtq_f32(values, upper_bound);

    // exp(x) = 2^n (1 + exp(r)), r = x - n*ln2, n = round(x / ln2)
    // exp(r) ~ poly(r) = r + r^2 * (c3 + c2 * r)
    float32x4_t n = vrndaq_f32(vmulq_f32(values, inv_ln2));
    float32x4_t r = vfmsq_n_f32(values, n, ln2);
    uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), 23);

    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t q = vfmaq_n_f32(c3, r, c2);
    float32x4_t s = vaddq_f32(vdupq_n_f32(1.0f), r);
    float32x4_t p = vfmaq_f32(s, q, r2);

    float32x4_t y =
        vreinterpretq_f32_u32(vaddq_u32(vreinterpretq_u32_f32(p), e));

    y = vbslq_f32(lt_lower, vdupq_n_f32(0.0f), y);
    y = vbslq_f32(gt_upper, vdupq_n_f32(INFINITY), y);
    return y;
  }

  // Defined out-of-line below, after the bitwise operators and fmadd() it
  // depends on.
  Vectorized erf() const;
};

inline Vectorized<float> operator+(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vaddq_f32(a, b);
}
inline Vectorized<float> operator-(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vsubq_f32(a, b);
}
inline Vectorized<float> operator*(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vmulq_f32(a, b);
}
inline Vectorized<float> operator/(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vdivq_f32(a, b);
}
inline Vectorized<float> operator&(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vreinterpretq_f32_u32(
      vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}
inline Vectorized<float> operator^(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vreinterpretq_f32_u32(
      veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

inline Vectorized<float> maximum(const Vectorized<float>& a,
                                 const Vectorized<float>& b) {
  return vmaxq_f32(a, b);
}
inline Vectorized<float> minimum(const Vectorized<float>& a,
                                 const Vectorized<float>& b) {
  return vminq_f32(a, b);
}
inline Vectorized<float> clamp(const Vectorized<float>& v,
                               const Vectorized<float>& min_v,
                               const Vectorized<float>& max_v) {
  return vminq_f32(vmaxq_f32(v, min_v), max_v);
}
inline Vectorized<float> fmadd(const Vectorized<float>& a,
                               const Vectorized<float>& b,
                               const Vectorized<float>& c) {
  return vfmaq_f32(c, a, b);
}

// Implementation adapted from ATen's NEON vec backend: Abramowitz & Stegun
// rational approximation, reusing exp_u20() for the exp(-x^2) term.
inline Vectorized<float> Vectorized<float>::erf() const {
  const Vectorized<float> neg_zero_vec(-0.f);
  const Vectorized<float> one_vec(1.0f);
  const Vectorized<float> p(0.3275911f);
  const Vectorized<float> p1(0.254829592f);
  const Vectorized<float> p2(-0.284496736f);
  const Vectorized<float> p3(1.421413741f);
  const Vectorized<float> p4(-1.453152027f);
  const Vectorized<float> p5(1.061405429f);
  // sign(x)
  auto sign_mask = neg_zero_vec & *this;
  auto abs_vec = this->abs();
  // t = 1 / (p * abs(x) + 1)
  auto tmp0 = fmadd(p, abs_vec, one_vec);
  auto t = one_vec / tmp0;
  // r = p5 * t^4 + p4 * t^3 + p3 * t^2 + p2 * t + p1
  auto tmp1 = fmadd(p5, t, p4);
  auto tmp2 = fmadd(tmp1, t, p3);
  auto tmp3 = fmadd(tmp2, t, p2);
  auto r = fmadd(tmp3, t, p1);
  // -exp(-x*x)
  auto pow_2 = (*this) * (*this);
  auto neg_pow_2 = pow_2 ^ neg_zero_vec;
  auto tmp4 = neg_pow_2.exp_u20();
  auto tmp5 = tmp4 ^ neg_zero_vec;
  // erf(x) = sign(x) * (1 - r * t * exp(-x*x))
  auto tmp6 = t * tmp5;
  auto tmp7 = fmadd(tmp6, r, one_vec);
  return tmp7 ^ sign_mask;
}

template <typename AccT>
inline AccT vec_reduce_add(const Vectorized<float>& v) {
  return static_cast<AccT>(vaddvq_f32(v));
}

template <>
struct Vectorized<c10::Half> {
  float16x8_t values;
  static constexpr int size_ = 8;

  Vectorized() : values(vdupq_n_f16(static_cast<float16_t>(0))) {}
  Vectorized(float16x8_t v) : values(v) {}
  explicit Vectorized(c10::Half v) {
    values = vdupq_n_f16(static_cast<float16_t>(static_cast<float>(v)));
  }

  static constexpr int size() { return size_; }

  operator float16x8_t() const { return values; }

  static Vectorized loadu(const c10::Half* ptr) {
    return vld1q_f16(reinterpret_cast<const float16_t*>(ptr));
  }
  static Vectorized loadu(const c10::Half* ptr, int count) {
    alignas(16) c10::Half buf[size_]{};
    copy_partial(buf, ptr, count);
    return loadu(buf);
  }

  void store(c10::Half* ptr) const {
    vst1q_f16(reinterpret_cast<float16_t*>(ptr), values);
  }
  void store(c10::Half* ptr, int count) const {
    alignas(16) c10::Half buf[size_];
    store(buf);
    copy_partial(ptr, buf, count);
  }

  template <typename Fn>
  Vectorized map_via_float(Fn fn) const {
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(values));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(values));
    alignas(16) float buf[8];
    vst1q_f32(buf, lo);
    vst1q_f32(buf + 4, hi);
    for (int i = 0; i < 8; ++i) {
      buf[i] = fn(buf[i]);
    }
    lo = vld1q_f32(buf);
    hi = vld1q_f32(buf + 4);
    return vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
  }

  Vectorized abs() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized(vabsq_f16(values));
#else
    return map_via_float([](float x) { return std::fabs(x); });
#endif
  }
  Vectorized sqrt() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized(vsqrtq_f16(values));
#else
    return map_via_float([](float x) { return std::sqrt(x); });
#endif
  }
};

inline Vectorized<c10::Half> convert_float_half(const Vectorized<float>& a,
                                                const Vectorized<float>& b) {
  return vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(b));
}

template <>
struct Vectorized<c10::BFloat16> {
  neon_bfloat16x8_t values;
  static constexpr int size_ = 8;

  Vectorized() : values{} {}
  Vectorized(neon_bfloat16x8_t v) : values(v) {}
  explicit Vectorized(c10::BFloat16 v) : values(bf16_dup(v)) {}

  static constexpr int size() { return size_; }

  operator neon_bfloat16x8_t() const { return values; }

  static Vectorized loadu(const c10::BFloat16* ptr) { return load_bf16x8(ptr); }
  static Vectorized loadu(const c10::BFloat16* ptr, int count) {
    alignas(16) c10::BFloat16 buf[size_]{};
    copy_partial(buf, ptr, count);
    return loadu(buf);
  }

  void store(c10::BFloat16* ptr) const { store_bf16x8(ptr, values); }
  void store(c10::BFloat16* ptr, int count) const {
    alignas(16) c10::BFloat16 buf[size_];
    store(buf);
    copy_partial(ptr, buf, count);
  }
};

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
    const Vectorized<c10::BFloat16>& a) {
  neon_bfloat16x8_t x = a;
  return {Vectorized<float>(bf16x4_to_f32(bf16_low(x))),
          Vectorized<float>(bf16x4_to_f32(bf16_high(x)))};
}

inline Vectorized<c10::BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a, const Vectorized<float>& b) {
  return bf16_combine(f32_to_bf16x4(a), f32_to_bf16x4(b));
}

}  // namespace vec_op::arm_vec
