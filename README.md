# zblas

A high-performance pure Zig BLAS (Basic Linear Algebra Subprograms) library. No C dependencies, no assembly — just Zig's `@Vector` and LLVM's backend doing the heavy lifting.

Built for AI inference workloads. Achieves **100-115 GFLOPS** (SGEMM) on x86-64, competitive with OpenBLAS.

## Performance

Benchmarked on x86-64 (AVX2):

| Matrix Size | GFLOPS | vs OpenBLAS |
|-------------|--------|-------------|
| 64x64       | ~87    | -           |
| 256x256     | ~106   | ~95%        |
| 512x512     | ~98    | ~90%        |
| 1024x1024   | ~107   | ~92%        |
| 3072x3072   | ~102   | ~93%        |

ARM64 (NEON) on Apple Silicon: up to **48 GFLOPS** (96% efficiency).

## Features

- **SGEMM** — Cache-blocked GotoBLAS-style algorithm with architecture-specific micro-kernels
  - x86-64: 8x8 AVX2 kernel with FMA, prefetching, 4x loop unrolling
  - ARM64: 8x8 NEON kernel with 16 Vec4 accumulators
  - Generic: 4x4 portable kernel using `@Vector`
- **SGEMV** — SIMD-optimized matrix-vector multiply
- **Level 1** — SAXPY, SDOT, SSCAL (reference implementations)
- **Parallel SGEMM** — Automatic M-dimension partitioning with size thresholds
- **Smart algorithm selection** — Direct SIMD for small matrices, cache-blocked with packing for large ones

## Quick Start

```zig
const zblas = @import("zblas");

// C = A * B  (row-major, MxK * KxN = MxN)
zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);

// C = alpha*A*B + beta*C
zblas.sgemm(M, N, K, A, B, C, alpha, beta);

// y = A*x
zblas.sgemv(M, N, A, x, y, 1.0, 0.0);
```

## As a Zig Package

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .zblas = .{
        .url = "https://github.com/tomoul/zblas/archive/refs/heads/main.tar.gz",
        .hash = "...",  // zig build will tell you the correct hash
    },
},
```

In your `build.zig`:

```zig
const zblas_dep = b.dependency("zblas", .{
    .target = target,
    .optimize = optimize,
});
your_module.addImport("zblas", zblas_dep.module("zblas"));
```

## Building

Requires **Zig 0.14+**.

```bash
# Run tests
zig build test

# Build and run benchmark
zig build bench && ./zig-out/bin/zblas-benchmark

# Build static library
zig build

# Cross-compile for ARM64
zig build -Dtarget=aarch64-linux
```

## API

### Level 3 (Matrix-Matrix)

```zig
// SGEMM: C = alpha*A*B + beta*C
zblas.sgemm(M, N, K, A, B, C, alpha, beta);

// With transpose options
zblas.sgemmTranspose(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);

// Parallel (auto thread count)
try zblas.sgemmParallel(allocator, M, N, K, A, B, C, alpha, beta);
```

### Level 2 (Matrix-Vector)

```zig
// SGEMV: y = alpha*A*x + beta*y
zblas.sgemv(M, N, A, x, y, alpha, beta);
```

### Level 1 (Vector)

```zig
zblas.saxpy(n, alpha, x, y);       // y = alpha*x + y
const dot = zblas.sdot(n, x, y);   // dot product
zblas.sscal(n, alpha, x);          // x = alpha*x
```

### Context API (for threading)

```zig
var ctx = zblas.Context.init(allocator, num_threads);
try ctx.sgemm(M, N, K, A, B, C, 1.0, 0.0);
```

## Architecture

```
src/
  zblas.zig              # Public API
  config.zig             # Compile-time tuning parameters
  reference.zig          # Naive reference implementations
  level3/
    sgemm.zig            # SGEMM dispatcher + algorithm selection
    sgemm_parallel.zig   # Multi-threaded SGEMM
  level2/
    sgemv.zig            # SGEMV implementation
  kernel/
    generic/sgemm_kernel_4x4.zig    # Portable 4x4 micro-kernel
    x86_64/sgemm_kernel_8x8.zig     # AVX2 8x8 micro-kernel
    arm64/sgemm_kernel_8x8.zig      # NEON 8x8 micro-kernel
  util/
    packing.zig           # Matrix packing (generic + arch dispatch)
    packing_x86_64.zig    # AVX2 packing with Vec8 loads
    packing_arm64.zig     # NEON packing with Vec4 loads
```

The SGEMM implementation follows the [GotoBLAS algorithm](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf):

1. **Tile** the matrices into MC x KC x NC blocks
2. **Pack** A and B into contiguous, cache-friendly layouts
3. **Dispatch** to architecture-specific micro-kernels
4. Smart **algorithm selection**: direct SIMD for small matrices, cache-blocked for large ones

## License

MIT
