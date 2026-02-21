<!--
SPDX-FileCopyrightText: 2025 CSC - IT Center for Science Ltd. <www.csc.fi>

SPDX-License-Identifier: CC-BY-4.0
-->

# Performance Portable C++ Programming with SYCL — Part II

In [Part I](ppp_with_sycl.parti_I.md) we introduced the core SYCL concepts: queues, kernels, memory models, and basic kernel launches, illustrated with a complete AXPY example using both the Buffer–Accessor and USM memory models.

In this second part we focus on practical topics that matter when writing real SYCL applications: how to use local shared memory for performance, how to express fine-grained dependencies between operations, how to perform basic profiling, and how to handle runtime errors robustly.

---

## Local Shared Memory

On GPUs, accesses to global memory are relatively slow compared to on-chip memory. Most GPU architectures provide a fast memory region that is shared among all threads in the same work-group. In CUDA and HIP this is called **shared memory**; in SYCL the equivalent is called **local memory**.

Local memory is visible to all work-items within a work-group, but not across different work-groups. It is typically used to:

- reduce global memory traffic,
- reuse data fetched from global memory across multiple threads,
- implement tiling patterns or parallel reductions.

Local memory is only available when kernels are launched using the **nd-range** method, since it requires an explicit notion of work-groups.

### Declaring and Using Local Memory

Local memory is declared inside a command group using a `local_accessor`:

```cpp
q.submit([&](handler& h) {

  local_accessor<float, 1> local_mem(range<1>(256), h);

  h.parallel_for(
    nd_range<1>(range<1>(N), range<1>(256)),
    [=](nd_item<1> item) {

      size_t lid = item.get_local_id(0);
      size_t gid = item.get_global_id(0);

      // Each work-item loads its element from global into local memory
      local_mem[lid] = d_x[gid];

      // Synchronize: ensure all work-items have finished writing before any reads
      item.barrier(access::fence_space::local_space);

      // Now it is safe to read from local memory
      d_y[gid] = a * local_mem[lid] + d_y[gid];
    });
});
```

Key points to remember:

- `local_accessor` allocates memory **per work-group**, not globally. Each work-group gets its own independent copy.
- Every work-item in the group can read and write this memory.
- A **barrier** is required whenever data written by one work-item is read by another. Without it, threads may read stale or uninitialized values, leading to race conditions and incorrect results.

---

## SYCL Dependencies

Because all queue operations are asynchronous, memory copies and kernel launches may overlap or execute in a different order than they were submitted — potentially producing incorrect results if dependencies are not handled properly.

There are three main approaches to managing dependencies:

1. **Coarse-grained synchronization** using `q.wait()`
2. **In-order queues** for sequential execution
3. **Fine-grained synchronization** using events

![Dependency graph illustration](img/graphs.svg)

### 1. Coarse-Grained Synchronization with `q.wait()`

The simplest approach is to call `q.wait()` after every operation that must complete before the next one begins:

```cpp
// Out-of-order queue (default)
queue q(default_selector_v);

constexpr size_t N = 25600;
std::vector<int> x(N), y(N);
int a = 4;

int* d_x = malloc_device<int>(N, q);
int* d_y = malloc_device<int>(N, q);

// Initialize d_x on device
q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) { d_x[i] = 1; });
});
q.wait(); // Block host until d_x initialization is complete

// Initialize d_y on device
q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) { d_y[i] = 2; });
});
q.wait(); // Block host until d_y initialization is complete

// Compute y = a * x + y
q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) {
    d_y[i] = a * d_x[i] + d_y[i];
  });
});
q.wait();

q.memcpy(y.data(), d_y, N * sizeof(int));
q.wait();

sycl::free(d_x, q);
sycl::free(d_y, q);
```

This guarantees correctness, but each `q.wait()` blocks the host until the entire queue drains. Operations that could safely overlap — such as initializing `d_x` and `d_y` simultaneously — are forced to run sequentially, which limits performance.

### 2. In-Order Queues

An `in_order` queue guarantees that submitted operations execute in the order they are submitted, without any explicit synchronization calls between them:

```cpp
// In-order queue — operations execute sequentially by construction
queue q(default_selector_v, property::queue::in_order{});

constexpr size_t N = 25600;
std::vector<int> x(N), y(N);
int a = 4;

int* d_x = malloc_device<int>(N, q);
int* d_y = malloc_device<int>(N, q);

// These three kernels execute in submission order — no wait() needed between them
q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) { d_x[i] = 1; });
});

q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) { d_y[i] = 2; });
});

q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) {
    d_y[i] = a * d_x[i] + d_y[i];
  });
});

// One wait at the end, after the final memcpy
q.memcpy(y.data(), d_y, N * sizeof(int));
q.wait();

sycl::free(d_x, q);
sycl::free(d_y, q);
```

This is clean and easy to reason about, and requires only a single `wait()` at the end. The trade-off is that operations which could safely run concurrently — such as the two initialization kernels — are serialized.

### 3. Fine-Grained Synchronization with Events (USM)

For the best performance, use **events**. Each queue operation returns an event, and subsequent operations can declare that they depend on specific events. This lets the runtime overlap independent operations while still respecting required ordering.

```cpp
// Out-of-order queue — allows concurrent execution where safe
queue q(default_selector_v);

constexpr size_t N = 25600;
std::vector<int> x(N), y(N);
int a = 4;

int* d_x = malloc_device<int>(N, q);
int* d_y = malloc_device<int>(N, q);

// These two kernels have no dependency on each other — they may run concurrently
auto event_x = q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) { d_x[i] = 1; });
});

auto event_y = q.submit([&](handler& h) {
  h.parallel_for(range<1>(N), [=](id<1> i) { d_y[i] = 2; });
});

// AXPY kernel depends on both initializations being complete
auto event_axpy = q.submit([&](handler& h) {
  h.depends_on({event_x, event_y}); // Explicit dependency declaration
  h.parallel_for(range<1>(N), [=](id<1> i) {
    d_y[i] = a * d_x[i] + d_y[i];
  });
});

// Memcpy depends on the AXPY kernel completing
auto event_copy = q.memcpy(y.data(), d_y, N * sizeof(int), {event_axpy});

// Only one host–device synchronization point: wait for the final memcpy
event_copy.wait();

sycl::free(d_x, q);
sycl::free(d_y, q);
```

The AXPY kernel will not start until both `event_x` and `event_y` are complete. The host-to-device copy will not start until AXPY finishes. Only a single `wait()` call is needed — right before the host accesses the results. This approach avoids unnecessary blocking, enables concurrent execution of independent operations, and generally yields the best performance.

### 4. Fine-Grained Synchronization with the Buffer–Accessor Model

When using the Buffer–Accessor model, fine-grained dependency management is largely **automatic**. The runtime tracks which kernels access which buffers and in what mode, and enforces the correct ordering without any explicit events.

```cpp
{
  buffer<int> x_buf(x.data(), range<1>(N));
  buffer<int> y_buf(y.data(), range<1>(N));

  // These two initialization kernels have no shared buffers — they may run concurrently
  q.submit([&](handler& h) {
    accessor x_acc(x_buf, h, write_only, no_init);
    h.parallel_for(range<1>(N), [=](id<1> i) { x_acc[i] = 1; });
  });

  q.submit([&](handler& h) {
    accessor y_acc(y_buf, h, write_only, no_init);
    h.parallel_for(range<1>(N), [=](id<1> i) { y_acc[i] = 2; });
  });

  // This kernel reads x_buf and writes y_buf.
  // The runtime automatically waits for both prior kernels to finish before launching it.
  q.submit([&](handler& h) {
    accessor x_acc(x_buf, h, read_only);
    accessor y_acc(y_buf, h, read_write);
    h.parallel_for(range<1>(N), [=](id<1> i) {
      y_acc[i] = a * x_acc[i] + y_acc[i];
    });
  });

  // host_accessor blocks the host until the kernel above is complete
  host_accessor y_acc(y_buf, read_only);
  bool passed = std::all_of(y_acc.begin(), y_acc.end(),
                            [a](int val) { return val == a * 1 + 2; });
  std::cout << (passed ? "SUCCESS" : "FAILURE") << std::endl;
}
```

The two initialization kernels can run concurrently since they access different buffers. The AXPY kernel is automatically held until both are done, because the runtime sees it requires read access to `x_buf` and `y_buf`, which were last written by the initialization kernels.

---

## Basic Profiling

Profiling is essential for understanding performance and identifying bottlenecks. SYCL provides built-in profiling support through a queue property.

### Enabling Profiling

To enable profiling, pass the `enable_profiling` property when creating the queue:

```cpp
queue q{ property::queue::enable_profiling{} };
```

### Measuring Kernel Execution Time

Every submitted operation returns an event. When profiling is enabled, that event stores timing information that can be queried after the operation completes:

```cpp
auto e = q.parallel_for(range<1>(N), [=](id<1> i) {
  d_y[i] = a * d_x[i] + d_y[i];
});

e.wait(); // Must wait before querying timing info

auto t_start = e.get_profiling_info<info::event_profiling::command_start>();
auto t_end   = e.get_profiling_info<info::event_profiling::command_end>();

double time_ms = (t_end - t_start) * 1e-6; // Values are in nanoseconds
std::cout << "Kernel time: " << time_ms << " ms\n";
```

> **Note:** Timing values are returned in nanoseconds. Some backends may have limited or no profiling support — check your implementation's documentation.

For deeper analysis, use the native vendor profiling tools:

- **Intel VTune** for oneAPI targets
- **NVIDIA Nsight** for CUDA backends
- **rocprof** for AMD GPUs

---

## Error Handling

SYCL's asynchronous execution model means that errors occurring on the device may surface well after the kernel was submitted. Standard C++ `try`/`catch` blocks alone are not sufficient — SYCL provides **asynchronous exception handlers** for this purpose.

### Asynchronous Exception Handler

An exception handler is a callable that processes errors the runtime captures during asynchronous execution. It is passed to the queue at construction time:

```cpp
auto async_handler = [](sycl::exception_list e_list) {
  for (auto& e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception& ex) {
      std::cerr << "Asynchronous SYCL exception: " << ex.what() << std::endl;
    }
  }
};

queue q{ default_selector_v, async_handler };
```

When an error occurs on the device, the runtime captures it, queues it internally, and passes it to this handler at the next synchronization point (such as a `wait()` call).

### Synchronous Exceptions

Some errors occur immediately and synchronously — for example, passing invalid arguments or requesting an unsupported allocation. These are raised as standard C++ exceptions and should be caught directly:

```cpp
try {
  int* d_x = malloc_device<int>(N, q);
} catch (sycl::exception const& e) {
  std::cerr << "SYCL exception: " << e.what() << std::endl;
}
```

### Best Practice: `wait_and_throw()`

When waiting for queue operations to complete, prefer `wait_and_throw()` over `wait()`:

```cpp
q.wait_and_throw(); // Waits AND flushes any pending asynchronous exceptions
```

This ensures that any errors that accumulated during asynchronous execution are immediately reported, rather than silently ignored.

---

## Summary

In this second part we covered several practical SYCL topics that are necessary for writing efficient and robust applications:

- **Local memory** enables fast data sharing within a work-group and is essential for memory-bound kernel optimization.
- **Dependency management** can be handled at three levels of granularity: coarse-grained `wait()`, in-order queues, or fine-grained events — each with different trade-offs between simplicity and performance.
- **Queue profiling** provides a lightweight way to measure kernel execution time directly from SYCL, complementing more advanced native profiling tools.
- **Asynchronous exception handling** is necessary for reliable error reporting in SYCL's non-blocking execution model.

These building blocks — combined with the concepts from Part I — provide a solid foundation for writing real SYCL applications. Future topics to explore include parallel reductions, subgroup operations, and performance tuning strategies.

---
## Resources and Further Reading

- [https://github.com/csc-training/portable-gpu-programming/tree/main](https://github.com/csc-training/portable-gpu-programming/tree/main)
- [https://github.com/csc-training/high-level-gpu-programming](https://github.com/csc-training/high-level-gpu-programming)
- [Data Parallel C++, second edition](https://link.springer.com/book/10.1007/978-1-4842-9691-2)
    - [book examples](https://github.com/Apress/data-parallel-CPP.git)

---