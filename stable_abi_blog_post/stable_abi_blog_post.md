### Introduction

vLLM relies heavily on PyTorch, with frequent calls to LibTorch in its C++ backend. As a result, changes to LibTorch’s binary interface may require vLLM to be rebuilt and relinked, even when the APIs remain unchanged. What if we could update LibTorch without rebuilding vLLM? That’s where the LibTorch Stable ABI comes in. With a LibTorch Stable ABI, updating LibTorch does not require rebuilding or updating vLLM in order to keep using it. Instead, we could swap out the LibTorch wheel and import it directly into our vLLM package. As of v0.33.0, vLLM publishes LibTorch ABI Stable CUDA wheels.  

<div style="text-align: center;">
<figure style="display: inline-block; width: 49%; margin: 0; vertical-align: top;">
<img src="images/unstable_import.png" alt="Unstable import" style="width: 100%; height: 250px; object-fit: contain;">
<figcaption>New torch requires vLLM to rebuild
</figcaption>
</figure>
<figure style="display: inline-block; width: 49%; margin: 0; vertical-align: top;">
<img src="images/stable_import.png" alt="Stable import" style="width: 100%; height: 250px; object-fit: contain;">
<figcaption>vLLM can import new torch directly
</figcaption>
</figure>
</div>

### What Is the LibTorch Stable ABI?
What is a Stable ABI? When a software library (such as LibTorch) is compiled into a binary, it exposes symbols that other programs can link against. The application binary interface (ABI) defines how these separately compiled components interact, including how functions are called, how data is represented in memory, and how symbols are named. Changes to these details can break binary compatibility, even when the source-level APIs remain unchanged. For example, changes to data layouts, function signatures, or C++ name mangling can prevent a program compiled against one version of a library from working with another. A stable ABI defines a set of binary interfaces and compatibility guarantees that allow programs to continue working across library updates without needing to be rebuilt, as long as they use interfaces covered by those guarantees. 

PyTorch has implemented its own LibTorch Stable ABI, which consists of three components. The first is a stable C shim that exposes a limited subset of LibTorch's functionality through a C API. This allows extensions to be built against one version of PyTorch and run against compatible versions without needing to be rebuilt. The second component is a set of C++ convenience wrappers, provided in header files, that build on top of the C shim. These wrappers provide higher-level C++ APIs instead of requiring developers to work directly with the lower-level C API. The third component is a collection of header-only utilities in the torch::headeronly namespace. These utilities are implemented entirely in header files and do not depend on libtorch.so, allowing them to be used without linking against LibTorch.

<img src="images/libtorch_stable.png" alt="LibTorch Stable" style="width: 100%; height: 250px; object-fit: contain;">





### Why Stable ABI Matters
Before adopting the LibTorch Stable ABI, vLLM often needed to rebuild its binaries to support a new PyTorch version. It currently takes vLLM over a month after a new PyTorch release to update its support for that version. As a result, downstream users may have to wait to take advantage of new PyTorch features through vLLM.
With the Stable ABI, users can upgrade to a compatible PyTorch version without rebuilding vLLM, provided the new version meets vLLM's compatibility requirements and the C++ and CUDA environments remain compatible. This allows users to adopt new PyTorch versions without waiting for a corresponding vLLM rebuild. For package distributors, the Stable ABI also decouples vLLM wheel releases from the PyTorch release cycle: a new PyTorch release no longer necessarily requires a new vLLM build, allowing the same vLLM binaries to work across multiple compatible PyTorch versions.


<div style="text-align: center;">
<figure style="display: inline-block; width: 49%; margin: 0; vertical-align: top;">
<img src="images/unstable_vllm.png" alt="Unstable ABI diagram" style="width: 100%; height: 250px; object-fit: contain;">
<figcaption>vLLM needs to change as LibTorch changes
</figcaption>
</figure>
<figure style="display: inline-block; width: 49%; margin: 0; vertical-align: top;">
<img src="images/stable_vllm.png" alt="Stable ABI diagram" style="width: 100%; height: 250px; object-fit: contain;">
<figcaption>vLLM can change independent of LibTorch
</figcaption>
</figure>
</div>


### Tradeoffs
Migrating vLLM to the Stable ABI comes with two main trade-offs: a more limited API surface and a small runtime overhead. Although the Stable ABI supports only a subset of LibTorch's APIs, it covers most of the APIs commonly used in vLLM, allowing us to migrate nearly all of its LibTorch API usage without workarounds. One exception is the method is_pinned(), which did not have a stable equivalent during the migration, so we had to replace
```cpp
if (cpu_tensor.is_pinned())
```
with 
```cpp
std::array<StableIValue, 2> is_pinned_stack{
     torch::stable::detail::from(cpu_tensor),
     torch::stable::detail::from(std::nullopt)};
 TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
     "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
 if (torch::stable::detail::to<bool>(is_pinned_stack[0]))
```
This was the only major API limitation for in-tree vLLM, and an is_pinned() stable method was added in torch 2.14. 
 
The runtime overhead comes primarily from dispatch through the PyTorch dispatcher and the boxing and unboxing of StableIValue arguments and return values. Dispatch overhead is on the order of a few microseconds and the overhead from boxing and unboxing is even smaller. For most kernels, this overhead is negligible relative to execution time and becomes significant only in rare cases of frequently launching very short-running kernels without optimizations such as CUDA graphs. We observed no noticeable runtime regressions from the migration.


<style>
.abi-story__row {
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(14rem, 1fr);
  column-gap: 2rem;
}
.abi-story__copy { grid-column: 1; grid-row: 1; min-width: 0; padding-bottom: 1.5rem; }
.abi-story__event { grid-column: 2; grid-row: 1; min-width: 0; border-left: 3px solid #6b7280; padding: 0 0 1.5rem 1rem; }
.abi-story__copy > :first-child, .abi-story__event > :first-child { margin-top: 0; }
.abi-story__copy > :last-child, .abi-story__event > :last-child { margin-bottom: 0; }
.abi-story__row:last-child > * { padding-bottom: 0; }
.abi-story__table { max-width: 100%; overflow-x: auto; }
@media (max-width: 48rem) {
  .abi-story__row { grid-template-columns: minmax(0, 1fr); }
  .abi-story__copy { padding-bottom: 0; }
  .abi-story__event { grid-column: 1; grid-row: 2; margin-bottom: 1.5rem; }
}
</style>

<div class="abi-story">
<div class="abi-story__row">
<div class="abi-story__copy">
<h3>Migrating vLLM to the Stable ABI</h3>
<p>We have broken down the migration of the Stable ABI into a series of steps: </p>
<p>1. <strong>Assess the situation</strong></p> 
<p> First, we need to do is figure out, can vLLM be migrate to the stable ABI using the current stable APIs. For this we took inventory of all the API calls to PyTorch in vLLM, made a list of APIs currently not supported by the Stable LibTorch, and made plans to add stable versions ops. This is outlined in the original <a href="https://github.com/vllm-project/vllm/issues/26946">RFC to make vLLM stable</a>.</p>
</div>
<div class="abi-story__event">
<h4>Timeline</h4>
<p><strong>October 15, 2025</strong><br>RFC opened to begin discussion of migrating to the Stable ABI. <a href="https://github.com/vllm-project/vllm/issues/26946">RFC</a></p>
</div>
</div>

<div class="abi-story__row">
<div class="abi-story__copy">
<p>2. <strong>Migrate AutogradFunctions and meta functions to Python</strong></p> 
<p> As of today, the LibTorch stable ABI does not support stable SymInt, so meta functions need to be written in Python instead of C++. Luckily, vLLM already had most of it's meta functions in Python so migrating the few that were not in there was relatively simple, and made the project more consistent. Similarly, the stable ABI does not expose PyTorch's autograd machinery, so AutogradFunctions also need to be written in Python. This does not matter as much for an inference engine like vLLM, but is worth noting that if a backwards function is implemented, it needs to be in Python.</p>
</div>
<div class="abi-story__event">
<p><strong>November 13, 2025</strong><br>Rewrite C++ meta funcs to Python. <a href="https://github.com/vllm-project/vllm/pull/28595">PR #28595</a></p>
</div>
</div>

<div class="abi-story__row">
<div class="abi-story__copy">
<p>3. <strong>Set up necessary scaffolding</strong></p> 
<p>Since vLLM is such a big project, with over 175 .cpp/.cu source files plus another 120+ header files in its CUDA build alone, we can't hope to migrate it all in one go. So instead we created a separate stable library (see CMakeLists.txt updates in the first migration <a href="https://github.com/vllm-project/vllm/pull/31509">PR #31509</a>), and we migrated the kernels to this library in manageable chunks.
<figure style="margin: 0; text-align: center;">
<img src="images/unstable_to_stable_library.png" alt="vLLM migration from the unstable library to the stable library" style="width: 100%; height: auto;">
<figcaption>As the stable library grows, the unstable library shrinks until it disappears</figcaption>
</figure>
</p>
</div>
<div class="abi-story__event">
<p><strong>December 29, 2025</strong><br>First migration PR opened: <a href="https://github.com/vllm-project/vllm/pull/31509">PR #31509</a></p>
<p><strong>March 19, 2026</strong><br>First migration PR closed.</p>
</div>
</div>



<div class="abi-story__row">
<div class="abi-story__copy">
<p>4. <strong>Make sure you register ops through the dispatcher</strong></p> 
<p> As of today, registering your LibTorch Stable ops requires registration through <code>STABLE_TORCH_LIBRARY</code>. This is very familiar if you already register your ops through <code>TORCH_LIBRARY</code>, which vLLM does. The biggest difference with the new stable registration is the that the implementation needs to be wrapped with <code>TORCH_BOX</code>. This generates an adapter that boxes and unboxes the C++ data types into the generic StableIValue types that can transport across PyTorch’s stable ABI.</p> 

<p>If your kernels are registered through a different means, such as pybind11, then it's recommended to first convert the registration to <code>TORCH_LIBRARY</code> before registering through <code>STABLE_TORCH_LIBRARY</code> (more on this below in the <strong>Challenges Along the Way</strong> section).</p>   

</div>
<div class="abi-story__event">
</div>
</div>

<div class="abi-story__row">
<div class="abi-story__copy">
<p>5. <strong>Migrate each source file to use stable APIs</strong></p> 
<p> Once everything in steps 1-5 have been set up, then we can start the migration to stable APIs. For this, we need to ensure that the APIs only come from torch headers included come from the C++ stable wrappers, the headeronly headers, or from the C-shim itself, i.e. they should come from one of these:
<ul>
<li><code>torch/csrc/stable</code></li>
<li><code>torch/headeronly</code></li>
<li><code>torch/csrc/stable/c/shim.h</code> (or <code>torch/csrc/inductor/aoti_torch/c/shim.h</code>) </li>
</ul>

The actual migration of the source files is mostly mechanical, e.g. convert 
<ul>
<li><code>torch::Tensor → torch::stable::Tensor</code></li>
<li><code>at::kInt → torch::headeronly::ScalarType::Int</code></li>
<li>etc. </li>
</ul>
To help migrate the source files, we created skills for AI agents that map existing APIs to their stable equivalents and flag common mistakes, such as changes unrelated to the migration. These skills also support the earlier steps (1–4) and while they have been useful in our work, we have not yet created evals to check them. (see skills in <a href="https://github.com/janeyx99/abi-stable-skills">abi-stable-skills</a>)

</p>

</div>
<div class="abi-story__event">
<p><strong>May, 2026</strong><br> Creation of <a href="https://github.com/janeyx99/abi-stable-skills">abi-stable-skills</a></p>
</div>
</div>

<div class="abi-story__row">
<div class="abi-story__copy">
<p>6. <strong>Verify!</strong></p> 
<p> vLLM is used by millions of users across multiple industries, and we do not want to interrupt any of their processes with our migration. Therefore, the most important step in this list is to <strong>Verify</strong> that the migration did not make any noticeable changes from before. This means running any appropriate unit and performance tests, but it also means manually going through each migration diff, line-by-line, and make sure there weren't any unintended changes.  

Along with verifying for correctness, we also want to verify that the new library is actually LibTorch ABI Stable. For this we have this very cool tool <a href="https://pypi.org/project/torch-abi-audit/">torch-abi-audit</a> (thanks Ralf Gommers!). This tool traces .so files and categorizes stable vs. unstable PyTorch symbols. If there aren't any unstable symbols, then the library is LibTorch ABI Stable. If the library is stable, it also tells you the min LibTorch version that has all of these stable symbols (e.g. <code>min-torch=2.11.0</code>). This tool was used a lot to check for the progress of the migration, check for regressions, and is now used in the vLLM CI to make sure the vLLM wheel stays LibTorch ABI Stable.  
   

</p>
<pre style="overflow-x: auto;"><code class="language-console">torch-abi-audit vllm 
Package: vllm
  Root: /workspaces/vllm_container/vllm/vllm
  <mark  style="background-color: #e74d50; color: #0f172a;">Torch ABI:   UNSTABLE </mark>
  Min torch:   2.11.0
  CPython ABI: mixed
  Extensions:  10
  Bundled libs: 0
  -- extensions --
    [STABLE  ] [abi3-ok               ] _C_stable_libtorch.abi3.so  (stable_shim=88, unstable=0)  min-torch=2.11.0
    [UNSTABLE] [abi3-ok               ] _flashmla_C.abi3.so  (stable_shim=0, unstable=70)
    [UNSTABLE] [abi3-ok               ] _flashmla_extension_C.abi3.so  (stable_shim=0, unstable=68)
    [STABLE  ] [abi3-ok               ] _moe_C_stable_libtorch.abi3.so  (stable_shim=72, unstable=0)  min-torch=2.11.0
    [UNSTABLE] [abi3-ok               ] _qutlass_C.abi3.so  (stable_shim=0, unstable=59)
    [NO-TORCH] [abi3-ok               ] cumem_allocator.abi3.so
    [NO-TORCH] [abi3-ok               ] spinloop.abi3.so
    [UNSTABLE] [uses-private-api      ] third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so  (stable_shim=0, unstable=59)
    [UNSTABLE] [abi3-ok               ] vllm_flash_attn/_vllm_fa2_C.abi3.so  (stable_shim=0, unstable=85)
    [UNSTABLE] [abi3-ok               ] vllm_flash_attn/_vllm_fa3_C.abi3.so  (stable_shim=0, unstable=81)
</code></pre>
torch-abi-audit before dependencies were migrated
</div>
<div class="abi-story__event">
<p><strong>March–June 19, 2026</strong><br>12+ more PRs were merged, making vLLM in-tree torch ABI Stable.</p>
</div>
</div>

<h3>vLLM build dependencies</h3>

<div class="abi-story__row">
<div class="abi-story__copy">
<p>Migrating vLLM source alone does not establish that the entire vLLM wheel is ABI-stable. For the wheel to be completely stable, its compiled dependencies must be torch ABI Stable as well. vLLM currently has seven dependencies that also depend on LibTorch. These are:</p>
<div class="abi-story__table">
<table>
<thead><tr><th>Dependency</th><th>Migrated on vLLM Fork</th><th>Migrated Upstream</th></tr></thead>
<tbody>
<tr><td>FlashMLA</td><td>✅</td><td>✅</td></tr>
<tr><td>DeepGEMM</td><td>✅</td><td>🚧 Work in progress</td></tr>
<tr><td>FlashAttention-2</td><td>✅</td><td>🚧 PR open to make it stable</td></tr>
<tr><td>FlashAttention-3</td><td>✅</td><td>✅</td></tr>
<tr><td>DeepSelect</td><td>✅</td><td>🚧 Work in progress</td></tr>
<tr><td>FlashKDA</td><td>✅</td><td>❌</td></tr>
<tr><td>QuTLASS</td><td>N/A</td><td>✅</td></tr>
</tbody>
</table>
</div>
<p>After the in-tree vLLM code had been migrated, each of these dependencies was migrated to the torch Stable ABI as well, making the entire vLLM wheel LibTorch ABI Stable.

<p>
<pre style="overflow-x: auto;"><code class="language-console">torch-abi-audit vllm
Package: vllm
  Root: /workspaces/vllm_container/vllm/vllm
  <mark  style="background-color: #4e7fc0; color: #0f172a;">Torch ABI:   STABLE </mark>
  Min torch:   2.11.0
  CPython ABI: abi3
  Extensions:  11
  Bundled libs: 0
  -- extensions --
    [STABLE  ] [abi3-ok               ] _C_stable_libtorch.abi3.so  (stable_shim=87, unstable=0)  min-torch=2.11.0
    [STABLE  ] [abi3-ok               ] _flashkda_C.abi3.so  (stable_shim=43, unstable=0)  min-torch=2.11.0
    [STABLE  ] [abi3-ok               ] _flashmla_C.abi3.so  (stable_shim=66, unstable=0)  min-torch=2.11.0
    [STABLE  ] [abi3-ok               ] _flashmla_extension_C.abi3.so  (stable_shim=63, unstable=0)  min-torch=2.11.0
    [STABLE  ] [abi3-ok               ] _moe_C_stable_libtorch.abi3.so  (stable_shim=72, unstable=0)  min-torch=2.11.0
    [NO-TORCH] [abi3-ok               ] cumem_allocator.abi3.so
    [NO-TORCH] [abi3-ok               ] fs_io_C.abi3.so
    [NO-TORCH] [abi3-ok               ] spinloop.abi3.so
    [STABLE  ] [abi3-ok               ] third_party/deep_gemm/_C_extension.abi3.so  (stable_shim=85, unstable=0)  min-torch=2.11.0
    [STABLE  ] [abi3-ok               ] vllm_flash_attn/_vllm_fa2_C.abi3.so  (stable_shim=69, unstable=0)  min-torch=2.10.0
    [STABLE  ] [abi3-ok               ] vllm_flash_attn/_vllm_fa3_C.abi3.so  (stable_shim=63, unstable=0)  min-torch=2.10.0
</code></pre>
torch-abi-audit after dependencies were migrated
</p>

</div>
<div class="abi-story__event">
<p><strong>July–September 2026</strong><br> FlashMLA, QuTLASS, FlashAttention 2, FlashKDA, FlashAttention 3, and DeepSelect move to the Stable ABI.</p>
</div>
</div>

<div class="abi-story__row">
<div class="abi-story__copy">
<h3>Challenges Along the Way</h3>
<p>
The biggest challenge was keeping pace with a large, rapidly changing project. Upstream files often changed while our migration patches were in progress, so we had to update those patches before they could merge. vLLM’s external dependencies also meant coordinating across projects and building support for the migration. At the API level, we sometimes needed wrappers for functions without Stable ABI equivalents or for operator arguments with unsupported types, such as using <code>int</code> instead of <code>int64_t</code>.

</div>
<div class="abi-story__event">
</div>
</div>



<div class="abi-story__row">
<div class="abi-story__copy">


DeepGEMM illustrates all of these challenges in one migration, along with several additional complications. Migrating it required us to:

<ul>
<li>Replace pybind11 registration with <code>TORCH_LIBRARY</code></li>
<li>Write wrappers for APIs without direct Stable ABI equivalents</li>
<li>Add wrappers where the Stable ABI requires a different operator argument types</li>
<li>Keep pace with DeepGEMM’s move from version 1 to version 2, including migrating its new LibTorch-dependent dependency,<a href="https://github.com/deepseek-ai/DeepJIT">DeepJIT</a>
</li>
<li>Account for ongoing repository changes, including vLLM’s switch from upstream DeepGEMM to its own fork</li>
</ul>

</div>
<div class="abi-story__event">
<p><strong>July 27-29</strong><br>First PRs to migrate DeepGEMM off pybind and onto the stable ABI are opened.</p>
</div>
</div>


<div class="abi-story__row">
<div class="abi-story__copy">
Once we had addressed these issues, we migrated DeepGEMM, the last remaining vLLM dependency. With that final piece in place, the entire vLLM wheel was LibTorch ABI Stable! 

</div>
<div class="abi-story__event">
<p><strong>October 15, 2026</strong><br>DeepGEMM migrated to the stable ABI, making the vLLM CUDA wheel completely torch ABI Stable — exactly one year after the <a href="https://github.com/vllm-project/vllm/issues/26946">RFC</a> was opened.</p>
</div>
</div>


### Example run on newer torch
```python
import importlib
import torch
import vllm

print(f"PyTorch runtime: {torch.__version__}")
print(f"vLLM: {vllm.__version__}")

# This module contains vLLM operators registered with STABLE_TORCH_LIBRARY.
stable_extension = importlib.import_module("vllm._C_stable_libtorch")

x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(2, 8) - 8
out = torch.empty(2, 4, device="cuda", dtype=x.dtype)
torch.ops._C.silu_and_mul(out, x)
print(f"silu_and_mul output:\n{out}")
```
**Output**

Build and run vLLM with torch 2.14.1

```
python stable_abi_demo.py           
PyTorch runtime: 2.14.1+cu132
vLLM: 0.1.dev24235+gc9af9acc6.d20261030
silu_and_mul output:
tensor([[1.0731e-02, 1.9132e-02, 2.9671e-02, 3.3464e-02],
        [0.0000e+00, 3.6553e+00, 1.0570e+01, 2.0004e+01]], device='cuda:0')
```

Update the torch wheel to torch==2.15.0:
```
uv pip install torch==2.15 \
  --index-url https://download.pytorch.org/whl/cu132
```
Rerun the script:
```
python stable_abi_demo.py
PyTorch runtime: 2.15.0+cu132
vLLM: 0.1.dev24235+gc9af9acc6.d20261030
silu_and_mul output:
tensor([[1.0731e-02, 1.9132e-02, 2.9671e-02, 3.3464e-02],
        [0.0000e+00, 3.6553e+00, 1.0570e+01, 2.0004e+01]], device='cuda:0')
```
### Future work
Now that vLLM’s CUDA build is on the torch Stable ABI, our next steps are to make all of vLLM’s builds ABI Stable. This includes migrating the CPU and ROCm builds as well. These migrations are not as large as the CUDA migration, but are still important parts of the vLLM ecosystem. Fortunately, the ROCm build already shares many of the same source files as the CUDA build. The .cu files are converted to .hip files through a hipify process and are thus already torch ABI Stable. Only a few ROCm-specific files remain to be migrated to the torch Stable ABI.



- PyTorch upstream updates to get rid of wrappers (mostly in DeepGEMM)
- More tests to make sure everything is truly stable



### Acknowledgement to main contributors

- vLLM Community: Shengqi Chen (main coordinator), Jiangyun Zhu (DeepSelect fork migration), Thien Tran (FlashKDA fork migration), Roberto L. Castro (upstream QuTLASS migration), Kaichao You (roadmap), and all maintainers who have reviewed our PRs and gave suggestions
- PyTorch Community: Jane Xu (project lead, FlashMLA migration in both fork & upstream, abi-stable-skills project), Chris Leonard (major migration work), Mikayla Gawarecki (early-stage migration work for in-tree source), Ralf Gommers (torch-abi-audit project)
- DeepSeek: Chenggang Zhao, Yichen Yan, Shengyu Liu
