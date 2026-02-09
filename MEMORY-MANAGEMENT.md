# PyTorch Memory Management Guide

Comprehensive guide to tensor memory management, clearing, reusing, and detaching from computation graphs in PyTorch.

## Table of Contents

1. [Tensor Lifecycle and References](#tensor-lifecycle-and-references)
2. [Clearing Tensors: del vs None vs detach](#clearing-tensors)
3. [Computation Graph Management](#computation-graph-management)
4. [Tensor Reuse Patterns](#tensor-reuse-patterns)
5. [Double/Triple Buffering](#doubletripple-buffering)
6. [Memory Clearing Operations](#memory-clearing-operations)
7. [Best Practices by Scenario](#best-practices-by-scenario)

---

## Tensor Lifecycle and References

### Python Reference Counting
```python
# Tensor creation
x = torch.randn(1000, 1000)  # Ref count = 1

# Additional reference
y = x  # Ref count = 2

# Remove one reference
del x  # Ref count = 1, memory NOT freed (y still references)

# Remove last reference
del y  # Ref count = 0, memory eligible for GC
```

### PyTorch Memory Allocator
- **CUDA Allocator**: Caches freed memory for reuse (doesn't return to OS immediately)
- **CPU Allocator**: Uses Python's memory allocator (returns to OS via GC)

---

## Clearing Tensors

### Method 1: `del` (Reference Removal)
```python
# Remove Python reference to tensor
x = torch.randn(1000, 1000, device='cuda')
del x  # Memory CACHED by PyTorch, not freed to GPU

# Still cached, available for reuse
y = torch.randn(1000, 1000, device='cuda')  # Likely reuses x's memory
```

**When to use:**
- Simple reference cleanup
- Memory will be reused soon
- Don't need immediate memory release

**Limitations:**
- Doesn't free GPU memory to OS
- Doesn't help with memory fragmentation
- Ineffective if PyTorch cache is full

### Method 2: `tensor = None` (Explicit Nulling)
```python
# Explicitly set reference to None
x = torch.randn(1000, 1000, device='cuda')
x = None  # Equivalent to del x, but more explicit

# Better for clarity in code
outputs = model(inputs)
loss = criterion(outputs, targets)
outputs = None  # Clear intention to free outputs
```

**When to use:**
- Same as `del`, but more explicit
- Code readability (shows intent to free)
- Multiple references to same tensor

**Advantages over del:**
- Can assign to same variable name later
- More explicit in code review
- Preferred in class attributes (`self.cache = None`)

### Method 3: `torch.cuda.empty_cache()` (Force GPU Release)
```python
# Force PyTorch to release cached memory back to GPU
x = torch.randn(1000, 1000, device='cuda')
del x
torch.cuda.empty_cache()  # Returns cached memory to GPU

# Check memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**When to use:**
- Before measuring memory usage
- After large batch of operations
- When approaching OOM
- Periodic cleanup in long-running loops

**Cost:**
- ~10-50ms overhead
- Next allocation slower (can't reuse cache)
- Use sparingly (not every iteration)

### Method 4: `.detach()` (Remove from Computation Graph)
```python
# Detach from autograd graph, keep data
x = torch.randn(1000, 1000, requires_grad=True)
y = x * 2  # y has grad_fn

z = y.detach()  # z has same data, NO grad_fn
# x, y still in computation graph
# z independent, gradient won't flow through z

del x, y  # Graph freed, z remains
```

**When to use:**
- Extract values from computation graph
- Prevent gradient computation on intermediate values
- Store results without keeping graph

**Memory impact:**
- Doesn't free data memory
- Frees computation graph memory (~2x data size for gradients)
- Useful with `@torch.no_grad()` context

### Method 5: `.cpu()` (Move to CPU)
```python
# Move tensor from GPU to CPU, return GPU memory
x = torch.randn(1000, 1000, device='cuda')
x_cpu = x.cpu()  # Copy to CPU
del x  # Free GPU memory
torch.cuda.empty_cache()

# Later, move back if needed
x_gpu = x_cpu.to('cuda')
```

**When to use:**
- Long-term storage of results
- Reduce GPU memory pressure
- Accumulate results from many batches

**Cost:**
- PCIe transfer overhead (~10-100 GB/s)
- Blocking operation (CPU waits for GPU)
- Use for infrequent operations only

---

## Computation Graph Management

### Problem: Autograd Graph Retention
```python
# BAD: Accumulates computation graphs
losses = []
for i in range(1000):
    output = model(input)
    loss = criterion(output, target)
    losses.append(loss)  # KEEPS GRAPH for all 1000 iterations

# Memory: 1000 × (model outputs + intermediate activations)
```

### Solution 1: Extract Scalar with `.item()`
```python
# GOOD: Extract scalar, discard graph
losses = []
for i in range(1000):
    output = model(input)
    loss = criterion(output, target)
    losses.append(loss.item())  # Scalar only, graph freed

# Memory: 1000 × sizeof(float) = 4KB (vs GB with graph)
```

### Solution 2: Use `@torch.no_grad()` Context
```python
# Disable gradient computation entirely
@torch.no_grad()
def evaluate(model, data):
    for batch in data:
        output = model(batch)  # No graph created
        # ... evaluation logic ...

# OR use context manager
with torch.no_grad():
    output = model(input)
```

### Solution 3: `.detach()` for Partial Graph Retention
```python
# Keep some tensors in graph, detach others
for i in range(1000):
    output = model(input)
    loss = criterion(output, target)

    # Need output values but not gradient
    output_detached = output.detach()

    # Compute loss on detached output (no backward through model)
    auxiliary_loss = some_function(output_detached)
```

---

## Tensor Reuse Patterns

### Pattern 1: In-Place Operations (When Safe)
```python
# In-place modification (suffix with _)
x = torch.randn(1000, 1000)
x.add_(1.0)  # Modifies x in-place, no new allocation
x.mul_(2.0)  # Modifies x in-place

# WARNING: Don't use in-place during autograd
x = torch.randn(1000, requires_grad=True)
y = x * 2
# x.add_(1.0)  # ERROR: modifying tensor used in graph
```

**When safe:**
- Outside autograd context (`@torch.no_grad()`)
- On tensors not requiring gradients
- After `.detach()`

**Memory savings:**
- No temporary tensor allocation
- 2x memory reduction for operations

### Pattern 2: Preallocate and Reuse Buffers
```python
# BAD: Allocate new tensor each iteration
for i in range(1000):
    output = torch.zeros(batch_size, hidden_dim, device='cuda')
    # ... compute into output ...

# GOOD: Preallocate buffer, reuse
output_buffer = torch.zeros(batch_size, hidden_dim, device='cuda')
for i in range(1000):
    output_buffer.zero_()  # Clear to zeros in-place
    # ... compute into output_buffer ...

    # Or use fill_()
    output_buffer.fill_(0.0)
```

**When to use:**
- Fixed-size tensors in loops
- Temporary storage buffers
- Accumulation arrays

**Limitations:**
- Must have same size/dtype
- Not safe during autograd
- Requires explicit zeroing

### Pattern 3: Scatter/Gather Instead of Creating New
```python
# BAD: Create new tensor with indexing
indices = torch.randint(0, 10, (5,))
subset = full_tensor[indices]  # New allocation

# GOOD: Use index_select with preallocated buffer
subset_buffer = torch.empty(5, full_tensor.size(1))
torch.index_select(full_tensor, 0, indices, out=subset_buffer)
```

---

## Double/Triple Buffering

### Problem: I/O and Compute Blocking
```python
# BLOCKING: CPU loads data while GPU waits
for batch in dataloader:
    batch_gpu = batch.to('cuda')  # GPU waits for CPU I/O
    output = model(batch_gpu)      # CPU waits for GPU compute
    # GPU idle during I/O, CPU idle during compute
```

### Solution: Double Buffering (Ping-Pong)
```python
# Overlap I/O with compute using two buffers
class DoubleBufferedDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        iterator = iter(self.dataloader)

        # Preload first batch
        try:
            next_batch = next(iterator)
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            next_batch = next_batch.to(self.device, non_blocking=True)

        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch

            # Start loading next batch while GPU computes
            try:
                next_batch = next(iterator)
                with torch.cuda.stream(self.stream):
                    next_batch = next_batch.to(self.device, non_blocking=True)
            except StopIteration:
                yield batch
                break

            yield batch

# Usage
loader = DoubleBufferedDataLoader(dataloader, 'cuda')
for batch in loader:
    output = model(batch)  # GPU compute overlaps with next batch I/O
```

### Triple Buffering (Advanced)
```python
# Three buffers: loading, ready, processing
# Useful for very slow I/O or complex preprocessing

from queue import Queue
from threading import Thread

class TripleBufferedLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.queue = Queue(maxsize=2)  # 2 buffers ahead

    def _loader_worker(self):
        for batch in self.dataloader:
            batch_gpu = batch.to(self.device, non_blocking=True)
            self.queue.put(batch_gpu)
        self.queue.put(None)  # Sentinel

    def __iter__(self):
        worker = Thread(target=self._loader_worker, daemon=True)
        worker.start()

        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch
```

**When to use:**
- I/O time > compute time (bottlenecked on data loading)
- Distributed training with slow network
- Large batch preprocessing

---

## Memory Clearing Operations

### Operation Summary

| Operation | Memory Impact | Graph Impact | Use Case |
|-----------|---------------|--------------|----------|
| `del tensor` | Removes Python ref | None | Basic cleanup |
| `tensor = None` | Same as del | None | Explicit nulling |
| `tensor.detach()` | None (shares data) | Removes grad_fn | Extract from graph |
| `tensor.cpu()` | Frees GPU, alloc CPU | None | Move to CPU |
| `tensor.clone()` | Allocates new | New graph node | Deep copy |
| `tensor.detach_()` | None | In-place detach | Dangerous in autograd |
| `torch.cuda.empty_cache()` | Frees cached to OS | None | Periodic cleanup |
| `gc.collect()` | Python GC run | None | Force Python cleanup |

### Tensor Clearing Checklist

```python
# Maximum memory cleanup (overkill for most cases)
def aggressive_cleanup(tensor):
    """
    Aggressively free all memory associated with tensor.
    Use only when absolutely necessary (expensive).
    """
    # 1. Detach from computation graph
    if tensor.requires_grad:
        tensor = tensor.detach()

    # 2. Move to CPU (if on GPU)
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 3. Delete Python reference
    del tensor

    # 4. Force PyTorch cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Force Python GC
    import gc
    gc.collect()

# Typical cleanup (sufficient for 99% of cases)
def typical_cleanup(tensor):
    """
    Standard cleanup for most cases.
    Fast and effective.
    """
    # 1. Extract scalar if needed
    if tensor.numel() == 1:
        value = tensor.item()

    # 2. Delete reference
    del tensor

    # That's it! PyTorch cache will handle the rest
    return value if 'value' in locals() else None
```

---

## Best Practices by Scenario

### Scenario 1: Evaluation Loop (No Gradients)
```python
@torch.no_grad()
def evaluate(model, dataloader):
    results = []

    for batch in dataloader:
        # Move to device
        inputs = batch['input'].to('cuda')
        targets = batch['target'].to('cuda')

        # Forward pass (no graph)
        outputs = model(inputs)

        # Extract metrics (scalars only)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
        results.append(accuracy)

        # Explicit cleanup (optional, helps with large models)
        del inputs, targets, outputs

    # Periodic cache cleanup every N batches
    if batch_idx % 256 == 0:
        torch.cuda.empty_cache()

    return sum(results) / len(results)
```

### Scenario 2: Training Loop (With Gradients)
```python
def train(model, dataloader, optimizer):
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['input'].to('cuda')
        targets = batch['target'].to('cuda')

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Extract loss scalar (frees graph)
        loss_value = loss.item()

        # Clear intermediate tensors
        # DON'T delete inputs/targets/outputs before backward!
        del inputs, targets, outputs, loss

        # Periodic cleanup
        if batch_idx % 256 == 0:
            torch.cuda.empty_cache()
```

### Scenario 3: Accumulating Results (Detached)
```python
def accumulate_features(model, dataloader):
    all_features = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to('cuda')

            # Extract features
            features = model.extract_features(inputs)

            # Move to CPU for accumulation (frees GPU memory)
            features_cpu = features.cpu()
            all_features.append(features_cpu)

            # Clear GPU tensors
            del inputs, features

    # Concatenate on CPU
    return torch.cat(all_features, dim=0)
```

### Scenario 4: Large Batch Processing (Chunking)
```python
def process_large_batch(model, large_input, chunk_size=32):
    """
    Process input larger than GPU memory by chunking.
    Reuses output buffer to avoid repeated allocation.
    """
    num_samples = large_input.size(0)
    output_buffer = None

    with torch.no_grad():
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk = large_input[start:end].to('cuda')

            # Process chunk
            chunk_output = model(chunk)

            # Initialize buffer on first iteration
            if output_buffer is None:
                output_shape = (num_samples, *chunk_output.shape[1:])
                output_buffer = torch.empty(output_shape,
                                           dtype=chunk_output.dtype,
                           device='cpu')

            # Copy to CPU buffer (frees GPU memory)
            output_buffer[start:end] = chunk_output.cpu()

            # Clear GPU memory
            del chunk, chunk_output

    return output_buffer
```

---

## Common Mistakes

### Mistake 1: Deleting Before Backward
```python
# WRONG: Deletes tensors needed for backward
outputs = model(inputs)
loss = criterion(outputs, targets)
del outputs  # ERROR: outputs needed for loss.backward()
loss.backward()

# CORRECT: Delete after backward
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
del outputs, loss  # OK: backward done
```

### Mistake 2: Excessive Cache Clearing
```python
# WRONG: Clear cache every iteration (huge overhead)
for batch in dataloader:
    output = model(batch)
    torch.cuda.empty_cache()  # 50ms overhead × 1000 iters = 50s wasted

# CORRECT: Periodic clearing
for i, batch in enumerate(dataloader):
    output = model(batch)
    if i % 256 == 0:
        torch.cuda.empty_cache()  # 50ms × 4 = 200ms total
```

### Mistake 3: Forgetting `.item()` for Scalars
```python
# WRONG: Accumulates computation graphs
losses = []
for batch in dataloader:
    loss = criterion(model(batch), targets)
    losses.append(loss)  # Keeps full graph

# CORRECT: Extract scalar
losses = []
for batch in dataloader:
    loss = criterion(model(batch), targets)
    losses.append(loss.item())  # Only scalar, graph freed
```

### Mistake 4: In-Place During Autograd
```python
# WRONG: In-place modification in autograd context
x = torch.randn(10, requires_grad=True)
y = x + 1
x.add_(2)  # ERROR: x used in y's computation graph

# CORRECT: Create new tensor or use no_grad
with torch.no_grad():
    x.add_(2)  # OK: outside autograd
```

---

## Summary

**Quick Reference:**

| Goal | Method | Cost |
|------|--------|------|
| Remove Python reference | `del tensor` or `tensor = None` | Free |
| Detach from graph | `tensor.detach()` | Free |
| Extract scalar | `tensor.item()` | Free |
| Free GPU memory | `del tensor; torch.cuda.empty_cache()` | ~50ms |
| Move to CPU | `tensor.cpu()` | PCIe transfer |
| Force GC | `gc.collect()` | ~10-100ms |
| Reuse buffer | `tensor.zero_()` or `tensor.fill_()` | Free |
| Double buffer | Separate streams | Setup cost |

**General Rules:**
1. Use `@torch.no_grad()` for evaluation/inference
2. Extract `.item()` for scalar metrics
3. Delete large tensors after use (`del`)
4. Periodic cache clear every 256-512 iterations
5. Move to CPU for long-term storage
6. Reuse buffers when possible
7. Avoid in-place during autograd
8. Don't over-optimize (measure first)

**Config Integration:**
See `nanochat/eval_config.py` for tunable memory management parameters.
