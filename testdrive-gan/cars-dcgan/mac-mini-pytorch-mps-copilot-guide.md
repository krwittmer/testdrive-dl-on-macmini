# GitHub Copilot guide: enable Apple GPU acceleration in existing PyTorch notebooks

Target: Python Jupyter notebooks running locally on an Apple Silicon Mac mini, using Anaconda/conda and PyTorch.

Check requirements against the installed PyTorch release; hardware, macOS, operator, and precision support vary by release.

## 1. Purpose and scope

Use this document as implementation instructions for GitHub Copilot when modifying an existing `.ipynb` notebook. Enable Apple's GPU through PyTorch's Metal Performance Shaders (MPS) backend while preserving the notebook's scientific purpose and training behavior.

The central distinction is:

```python
device = torch.device("mps")  # Select a destination; this alone moves nothing.
model = model.to(device)      # Place model parameters and buffers on the GPU.
inputs = inputs.to(device)    # Place input tensors on the GPU.
outputs = model(inputs)       # Supported operations execute on the GPU.
```

Tensors can also be created directly on the GPU with `device=device`. Ordinary PyTorch code defaults to CPU unless placement is changed. Apple Silicon unified memory does not eliminate PyTorch's device rules. MPS targets the GPU, not the Apple Neural Engine. [1, 2]

Apple GPU cores are execution resources within a GPU, not individual PyTorch devices to which notebooks should be assigned. Use `mps`; do not invent a binding per GPU core. CUDA targets NVIDIA GPUs. Installing CUDA, cuDNN, or adding a `@Cuda` decorator will not enable an Apple GPU.

This guide applies to PyTorch portions of a notebook. It does not automatically accelerate NumPy, pandas, scikit-learn, TensorFlow, or custom CUDA extensions. Identify the actual framework before making changes. Report incompatible dependencies rather than silently replacing them.

## 2. Instructions to Copilot before editing

1. Inspect all notebook cells and imported local training helpers. Find model creation, optimizers, checkpoint loads, training, validation, inference, metrics, and plotting.
2. Identify every model, loss module with state, input, target, mask, index tensor, recurrent state, and temporary tensor participating in device computation.
3. Find hardcoded `.cuda()`, `device="cuda"`, `torch.cuda.*`, CUDA tensor constructors, and branches that choose CUDA or CPU while overlooking MPS. Adapt according to meaning; do not perform a blind string replacement.
4. Preserve model architecture, dataset splits, preprocessing, loss definitions, optimizer settings, random seeds, batch size, epoch count, gradient accumulation, and GAN update schedules unless a necessary compatibility change is explained.
5. Preserve integer indices and labels, Boolean masks, and complex types when required. Do not cast every tensor to floating point. If double precision is essential to the algorithm, report that this path must remain on CPU.
6. Preserve notebook cell order, useful markdown, cell IDs, and metadata. Keep the `.ipynb` valid. Clear outputs made stale by changed cells; do not fabricate execution results or timings.
7. Add one shared device configuration near the top. Reuse it throughout the notebook and affected helper functions.
8. Deliver the edited notebook and a short change/validation report. If a local MPS kernel is unavailable, state that GPU execution remains unverified.

## 3. Anaconda prerequisites and environment setup

The Jupyter kernel must execute on the Mac. A browser on the Mac connected to Colab, a remote server, or a Linux container does not automatically expose the Mac GPU to that kernel. Use native macOS Python for this workflow.

Use the Apple Silicon version of Anaconda and native `arm64` Python. If the existing environment already reports `arm64` and MPS availability, retain it when compatible with the notebook's dependencies. A working environment does not need to be rebuilt merely to use this guide.

For a fresh environment, run the following in macOS Terminal, not in Python cells:

```bash
conda create -n torch-mac python=3.12 pip ipykernel jupyterlab
conda activate torch-mac
python -c "import platform; print(platform.machine())"
```

Expect `arm64`. If it reports `x86_64`, investigate an Intel Anaconda installation, Rosetta, or an environment targeting `osx-64`. Use a native Apple Silicon environment. Do not try to repair an Intel environment simply by toggling the notebook's device string.

Install compatible stable PyTorch packages inside the activated environment:

```bash
python -m pip install torch torchvision
python -m ipykernel install --user --name torch-mac --display-name "Python (Anaconda PyTorch MPS)"
jupyter lab
```

Install `torchaudio` only if needed. Respect existing dependency pins; resolve compatibility deliberately rather than upgrading a working project indiscriminately. Official PyTorch conda publication ended starting with 2.6; official pip wheels can be installed inside conda environments. Separately maintained conda packages may also exist. Avoid installing competing pip and conda builds of PyTorch in the same environment. [3]

Select **Python (Anaconda PyTorch MPS)** in JupyterLab, Notebook, or VS Code's notebook kernel picker. Anaconda Navigator can launch Jupyter, but the selected kernel determines the Python and packages used. Restart the kernel after changing PyTorch packages. [4]

Check macOS/Python requirements for the chosen stable release. Apple's setup page also lists Xcode command-line tools; if needed, install them once in Terminal with `xcode-select --install`. Do not switch to nightly builds by default. [2]

## 4. Add an early device configuration cell

Use this default for notebooks whose objective is verified Mac GPU execution:

```python
import os
import platform
import sys
import torch

print("Python:", sys.executable)
print("Architecture:", platform.machine())
print("macOS:", platform.mac_ver()[0])
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("MPS CPU fallback:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"))

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is unavailable. Check the selected conda kernel, native arm64 "
        "Python, supported macOS version, and PyTorch installation."
    )

device = torch.device("mps")
print("Selected device:", device)
```

`is_built()` checks the PyTorch build; `is_available()` checks usability in the current environment. A printed device string alone does not demonstrate that a model uses the GPU. [1]

If cross-platform portability is explicitly required, replace the selection logic with an explicit preference. Do not add this alternative as a second competing configuration cell:

```python
DEVICE_PREFERENCE = "mps"  # "mps", "cuda", "cpu", or "auto"

if DEVICE_PREFERENCE == "auto":
    selected = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
else:
    selected = DEVICE_PREFERENCE

if selected not in {"mps", "cuda", "cpu"}:
    raise ValueError(f"Unsupported device preference: {selected}")
if selected == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS requested but unavailable")
if selected == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable")

device = torch.device(selected)
print("Selected device:", device)
```

Use explicit `.to(device)` calls rather than globally changing the default device. This keeps CPU preprocessing, data loading, and NumPy interoperability predictable.

## 5. Adapt model creation, optimizers, and tensor placement

The following are adaptation patterns, not replacements for the notebook's model or loss. Replace illustrative names with the existing objects.

```python
model = ExistingModel(...)
model = model.to(device)

# Construct the optimizer after moving the model.
optimizer = torch.optim.Adam(model.parameters(), lr=existing_learning_rate)
```

Retain the existing optimizer class and arguments. If the original model has double-precision floating parameters, convert to float32 only if appropriate for its numerical requirements. Move loss modules with state, including class weights, too:

```python
criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
```

Move each batch in the main training process. For classification using integer class IDs:

```python
inputs = inputs.to(device=device, dtype=torch.float32)
targets = targets.to(device=device, dtype=torch.long)
```

For regression or binary/multilabel targets, use the dtype and shape required by the existing loss, often float32. Token IDs and embedding indices usually remain `torch.long`; masks may be Boolean. MPS does not support float64, and transferring float64 tensors does not automatically downcast them. CPU fallback does not fix this limitation. [5]

For nested batches, retain their structure. Where useful, introduce a helper for standard containers:

```python
def move_to_device(value, device):
    """Move tensors in plain dict/list/tuple batches; preserve tensor dtypes."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value
```

This helper preserves dtypes; handle inappropriate float64 data separately at ingestion. Add type-specific handling for named tuples, dataclasses, or custom batch classes if present; do not apply this helper to such containers without adapting it.

Inside models and helpers, allocate temporary tensors beside the tensors they operate on:

```python
noise = torch.randn_like(inputs)
scratch = inputs.new_zeros((inputs.shape[0], hidden_size))
indices = torch.arange(inputs.shape[0], device=inputs.device)
```

Register persistent non-parameter tensor state using `register_buffer` when appropriate. Tensors hidden in ordinary attributes or plain lists are not necessarily moved by `model.to(device)`; inspect those cases.

## 6. Preserve the existing training and evaluation logic

For a standard classification loop, the resulting placement should resemble:

```python
model.train()
for inputs, targets in train_loader:
    inputs = inputs.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
```

Use `set_to_none=True` only where compatible with existing gradient handling. Preserve special gradient accumulation, clipping, skipped updates, and multiple-optimizer behavior.

For ordinary evaluation/inference:

```python
model.eval()
with torch.inference_mode():
    inputs = inputs.to(device=device, dtype=torch.float32)
    predictions = model(inputs)

predictions_numpy = predictions.detach().cpu().numpy()
```

Only use inference mode where gradients will not be needed. Preserve gradient-based evaluation such as saliency, adversarial examples, or gradient penalties. Return to `model.train()` before subsequent training.

For GANs, move both generator and discriminator, their inputs, labels, and latent noise. Construct both optimizers after placing the networks. Preserve existing alternating updates, `.detach()` boundaries, and gradient penalties:

```python
generator = generator.to(device)
discriminator = discriminator.to(device)
z = torch.randn(batch_size, latent_dim, device=device)
```

Do not detach generated samples in a generator update if doing so would break its gradient path.

## 7. DataLoader and Jupyter settings

Retain the original batch size initially. For a reliable notebook baseline:

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=existing_batch_size,
    shuffle=True,        # Preserve the notebook's original sampling behavior.
    num_workers=0,
    pin_memory=False,
)
```

Preserve custom samplers, collation, and other required arguments. Keep datasets and worker-side preprocessing on CPU; move batches to MPS in the main process. Do not copy CUDA-specific pinned-memory assumptions into the MPS configuration.

If loading is a bottleneck, benchmark a small worker count. macOS uses spawn for multiprocessing, so custom dataset classes and worker functions should be importable from `.py` modules; notebook-only definitions can fail. Use `persistent_workers=True` only with `num_workers > 0`. [6]

## 8. Unsupported operators and CUDA-only dependencies

Start with optional MPS CPU fallback disabled. Check inherited environment settings; do not claim an all-GPU run while fallback is enabled.

If an operator is unsupported, first identify the exact operation and check support in the installed stable release. If necessary, document a compatible implementation or a deliberate CPU segment. Do not catch every device error and rerun the entire workload on CPU silently.

Optional compatibility mode must be configured before importing PyTorch in a freshly restarted kernel, or when launching Jupyter:

```bash
conda activate torch-mac
PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter lab
```

Alternatively, the first notebook cell can contain:

```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Import torch only after this cell in a fresh kernel.
```

Fallback applies to eligible unsupported operations; it does not guarantee compatibility with every model, fix float64 allocation, or translate custom CUDA kernels. Document warnings and measure performance with fallback enabled. To disable an inherited setting, set it to `0` before importing PyTorch and restart the kernel. [5, 7]

Guard NVIDIA-only settings if maintaining a CUDA path. Review `.cuda()`, `torch.cuda.amp`, CUDA synchronization, cuDNN flags, fused optimizers, distributed wrappers, and third-party CUDA extensions individually. There is no universal MPS equivalent for each CUDA API.

## 9. Checkpoints

Make checkpoint loading independent of the original training device:

```python
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)  # For a plain model state_dict checkpoint.
model = model.to(device)
```

If the checkpoint contains a dictionary such as `checkpoint["model_state_dict"]`, use its actual structure. `weights_only=True` is appropriate for ordinary tensor state dictionaries; document other formats and load only trusted checkpoints. Do not disable loading safeguards merely to suppress an error.

For resumed training, restore the existing optimizer/scheduler/scaler state in the correct order after model placement. Inspect optimizer tensor state: parameter-shaped moments must be compatible with their parameters, while scalar bookkeeping can be optimizer-specific. Do not blindly move every optimizer tensor or recreate the optimizer after loading and thereby discard its state. Verify a real resumed optimizer step.

## 10. Performance tuning after correctness

Treat tuning as a separate, measured step:

- Compare end-to-end samples/second for the actual workload against a CPU baseline. Small models or batches can be faster on CPU because GPU dispatch has overhead.
- Tune batch size gradually; changes can affect optimization and convergence. Gradient accumulation increases effective batch size but does not make each small microbatch saturate the GPU.
- Reduce repeated `.item()`, `.cpu()`, and `.numpy()` calls in hot loops. Detach metrics before retaining them; retain the original metric weighting and aggregation semantics.
- Establish a float32 baseline. Consider mixed precision only after checking MPS support for the exact PyTorch/macOS/model combination and validating loss and gradients. Do not copy CUDA AMP or GradScaler settings unchanged, or call `.half()` indiscriminately. GANs may be especially sensitive.
- Keep default memory-watermark and fast-math settings initially. Do not disable the allocator's memory limit as a routine optimization. Do not call `torch.mps.empty_cache()` every batch. It releases unused cached memory, not live tensors. [7, 8]
- Run one training notebook at a time when benchmarking. Other kernels compete for GPU compute and unified memory. Shut down unused kernels; retained notebook variables can keep large tensors alive.
- Treat channels-last layout and compilation as optional experiments requiring release-specific support checks and measured benefit, not mandatory migration changes.

Use synchronized timing. This helper works with the optional cross-platform configuration:

```python
import time

def synchronize_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)

# benchmark_inputs is a representative batch already placed on device.
was_training = model.training
model.eval()
with torch.inference_mode():
    for _ in range(5):
        _ = model(benchmark_inputs)
    synchronize_device(device)

    start = time.perf_counter()
    for _ in range(20):
        _ = model(benchmark_inputs)
    synchronize_device(device)
    seconds_per_batch = (time.perf_counter() - start) / 20

model.train(was_training)
print(f"Inference seconds/batch: {seconds_per_batch:.6f}")
```

Adapt model arguments for multi-input models. This measures inference on an already placed batch; it excludes data loading and training. Training benchmarks must include forward, backward, and optimizer steps and use disposable model/optimizer state so measurement does not unintentionally train the user's model. GPU work is asynchronous; synchronization is required at timing boundaries. [8]

## 11. Validation and acceptance checklist

Copilot should complete the checks it can run and explicitly mark the rest as pending:

- [ ] The notebook remains valid and retains its original task, architecture, data splits, and training semantics.
- [ ] A fresh kernel can execute cells in order without relying on stale notebook state.
- [ ] Kernel diagnostics show the intended conda Python, `arm64`, and MPS availability.
- [ ] One MPS tensor operation actually executes and synchronizes successfully.
- [ ] Model parameters/buffers, representative inputs, relevant targets, and outputs have the expected devices and dtypes.
- [ ] A representative forward pass, finite loss, backward pass, and optimizer step complete; relevant gradients are finite.
- [ ] For GANs, both generator and discriminator update paths complete.
- [ ] Validation/inference and conversion to plotting/NumPy formats work.
- [ ] Checkpoint save/reload works; resumed training is tested if used.
- [ ] Unsupported operations and CPU fallback settings are recorded rather than hidden.
- [ ] Performance claims use measured synchronized timings and identify hardware, software, batch size, and precision.
- [ ] If no Apple GPU is available during editing, report static checks separately and leave GPU execution/performance unverified.

Use a small representative batch for initial checks. Run smoke-training steps on disposable state or restore all affected state afterward; do not silently advance the real training run. For numerical comparison, use the same initial weights and inputs on CPU and MPS with reasonable tolerances. Do not require bitwise equality across devices.

Suggested placement diagnostics after a forward pass:

```python
print("Parameter devices:", {str(p.device) for p in model.parameters()})
print("Buffer devices:", {str(b.device) for b in model.buffers()})
print("Inputs:", inputs.device, inputs.dtype)
print("Outputs:", predictions.device, predictions.dtype)
```

Adapt diagnostics to nested inputs and outputs. Device placement confirms where tensors live; it does not by itself prove every operation avoided CPU fallback or that GPU execution is faster.

## 12. Prompt to accompany this file in Copilot

> Use `mac-mini-pytorch-mps-copilot-guide.md` to update the selected Python Jupyter notebook for local Apple Silicon Mac mini GPU execution using Anaconda and PyTorch MPS. Inspect the entire notebook and relevant local helpers first. Implement one early device configuration, move every participating model and tensor appropriately, preserve dtype and training semantics, and adapt CUDA-specific assumptions deliberately. Keep the existing architecture, loss, data splits, optimizer configuration, and hyperparameters. Use explicit MPS selection with a visible error if unavailable, unless I request cross-platform fallback. Add concise setup instructions and representative placement/execution checks. Keep optional performance experiments separate from the initial migration. Return the modified notebook and summarize changes, completed validation, remaining compatibility issues, and any checks requiring my Mac GPU. Do not claim GPU execution or speedups without running the corresponding checks.

## 13. Official references

1. [PyTorch MPS backend and device placement](https://docs.pytorch.org/docs/stable/notes/mps.html)
2. [Apple: accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
3. [PyTorch 2.6: official conda publication change](https://pytorch.org/blog/pytorch2-6/)
4. [IPython: installing kernels for conda environments](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)
5. [PyTorch MPS notes: double-precision limitation](https://docs.pytorch.org/docs/main/notes/mps.html) — this link is development documentation; verify other features against your installed stable release.
6. [PyTorch DataLoader and platform-specific multiprocessing](https://docs.pytorch.org/docs/stable/data.html)
7. [PyTorch MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
8. [PyTorch MPS synchronization, memory, and profiling APIs](https://docs.pytorch.org/docs/stable/mps.html)

Where examples contain illustrative names such as `ExistingModel`, `existing_batch_size`, or `benchmark_inputs`, adapt them to the notebook's real objects. They are not additional dependencies or executable cells to insert verbatim.

## Suggested order of fixes

1. Set both learning rates to 2e-4, add the paper's weight initialization, and fix the discriminator (no BatchNorm on the first layer, 4×4 kernels, widths 64→512, a conv head).
2. Add monitoring: fixed noise, an image grid, `D(x)` and `D(G(z))` logging, and a seed.
3. Fix the data: resize plus center-crop, a horizontal flip, `.convert('RGB')`, and cache the resized tensors.
4. Reuse one fake batch per step.
5. Optional: try the LSGAN loss, spectral norm, or DiffAugment.
