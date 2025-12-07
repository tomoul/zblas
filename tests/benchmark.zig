// zblas/tests/benchmark.zig
// Performance benchmarks for zblas

const std = @import("std");
const zblas = @import("zblas");

const print = std.debug.print;

pub fn main() !void {
    print("\n", .{});
    print("===========================================\n", .{});
    print("           zblas Benchmark Suite           \n", .{});
    print("===========================================\n\n", .{});

    // Print system info
    const info = zblas.getInfo();
    print("Architecture: {s}\n", .{info.arch});
    print("SIMD enabled: {}\n", .{info.simd});
    print("Vector width: {} floats\n", .{info.vector_width});
    print("Micro-kernel: {}x{}\n", .{ info.mr, info.nr });
    print("CPU cores: {}\n", .{std.Thread.getCpuCount() catch 1});
    print("\n", .{});

    // Run SGEMM benchmarks
    try benchmarkSgemm();

    // Run parallel SGEMM scaling benchmarks
    try benchmarkParallelScaling();

    // Run SGEMV benchmarks
    try benchmarkSgemv();

    print("\n===========================================\n", .{});
    print("           Benchmark Complete              \n", .{});
    print("===========================================\n", .{});
}

fn benchmarkSgemm() !void {
    print("SGEMM Benchmarks (C = A * B)\n", .{});
    print("-------------------------------------------\n", .{});
    print("{s:>8} {s:>12} {s:>12} {s:>10}\n", .{ "Size", "Time (ms)", "GFLOPS", "Efficiency" });
    print("-------------------------------------------\n", .{});

    const sizes = [_]usize{ 64, 128, 256, 512, 1024 };
    const allocator = std.heap.page_allocator;

    for (sizes) |n| {
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(n)) *
            @as(f64, @floatFromInt(n)) *
            @as(f64, @floatFromInt(n));

        // Allocate matrices
        const A = try allocator.alloc(f32, n * n);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, n * n);
        defer allocator.free(B);
        const C = try allocator.alloc(f32, n * n);
        defer allocator.free(C);

        // Initialize with random-ish values
        for (0..n * n) |i| {
            A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
            B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
        }
        @memset(C, 0.0);

        // Warmup
        zblas.sgemm(n, n, n, A, B, C, 1.0, 0.0);

        // Benchmark
        const iterations: usize = if (n <= 256) 10 else if (n <= 512) 5 else 3;
        var total_time: f64 = 0.0;

        for (0..iterations) |_| {
            @memset(C, 0.0);

            var timer = std.time.Timer.start() catch unreachable;
            zblas.sgemm(n, n, n, A, B, C, 1.0, 0.0);
            const elapsed = timer.read();

            total_time += @as(f64, @floatFromInt(elapsed)) / 1_000_000.0; // ns to ms
        }

        const avg_time = total_time / @as(f64, @floatFromInt(iterations));
        const gflops = flops / (avg_time / 1000.0) / 1e9;

        // Theoretical peak (rough estimate based on architecture)
        const peak_gflops: f64 = switch (@import("builtin").cpu.arch) {
            .x86_64 => 100.0, // ~100 GFLOPS single-threaded AVX2
            .aarch64 => 50.0, // ~50 GFLOPS Apple M1/M2
            else => 10.0,
        };
        const efficiency = gflops / peak_gflops * 100.0;

        print("{d:>8} {d:>12.3} {d:>12.2} {d:>9.1}%\n", .{
            n,
            avg_time,
            gflops,
            efficiency,
        });
    }

    print("\n", .{});
}

fn benchmarkParallelScaling() !void {
    print("Parallel SGEMM Scaling Test\n", .{});
    print("-------------------------------------------\n", .{});
    print("Matrix size: 1024x1024 (large enough for parallelization)\n", .{});
    print("-------------------------------------------\n", .{});
    print("{s:>8} {s:>12} {s:>12} {s:>10}\n", .{ "Threads", "Time (ms)", "GFLOPS", "Speedup" });
    print("-------------------------------------------\n", .{});

    const n: usize = 1024;
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(n)) *
        @as(f64, @floatFromInt(n)) *
        @as(f64, @floatFromInt(n));

    const allocator = std.heap.page_allocator;

    // Allocate matrices
    const A = try allocator.alloc(f32, n * n);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, n * n);
    defer allocator.free(B);
    const C = try allocator.alloc(f32, n * n);
    defer allocator.free(C);

    // Initialize
    for (0..n * n) |i| {
        A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    }

    const cpu_count = std.Thread.getCpuCount() catch 1;
    const max_threads = @min(cpu_count, 8);
    const thread_counts = [_]usize{ 1, 2, 4, 8 };
    var baseline_time: f64 = 0.0;

    for (thread_counts) |num_threads| {
        if (num_threads > max_threads) continue;

        @memset(C, 0.0);

        // Warmup
        zblas.sgemmParallelN(allocator, num_threads, n, n, n, A, B, C, 1.0, 0.0) catch {};

        // Benchmark
        const iterations: usize = 3;
        var total_time: f64 = 0.0;

        for (0..iterations) |_| {
            @memset(C, 0.0);

            var timer = std.time.Timer.start() catch unreachable;
            zblas.sgemmParallelN(allocator, num_threads, n, n, n, A, B, C, 1.0, 0.0) catch {};
            const elapsed = timer.read();

            total_time += @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
        }

        const avg_time = total_time / @as(f64, @floatFromInt(iterations));
        const gflops = flops / (avg_time / 1000.0) / 1e9;

        if (num_threads == 1) baseline_time = avg_time;
        const speedup = if (baseline_time > 0) baseline_time / avg_time else 1.0;

        print("{d:>8} {d:>12.3} {d:>12.2} {d:>9.2}x\n", .{
            num_threads,
            avg_time,
            gflops,
            speedup,
        });
    }

    // Also test automatic thread selection
    print("-------------------------------------------\n", .{});
    print("Auto-select threads test:\n", .{});

    const optimal_threads = zblas.getOptimalThreadCount(n, n, n);
    print("  Optimal threads for {}x{}: {}\n", .{ n, n, optimal_threads });
    print("  shouldParallelize({}x{}): {}\n", .{ n, n, zblas.shouldParallelize(n, n, n) });

    // Test with small matrix (should NOT parallelize)
    const small_n: usize = 128;
    print("  shouldParallelize({}x{}): {}\n", .{ small_n, small_n, zblas.shouldParallelize(small_n, small_n, small_n) });

    print("\n", .{});
}

fn benchmarkSgemv() !void {
    print("SGEMV Benchmarks (y = A * x)\n", .{});
    print("-------------------------------------------\n", .{});
    print("{s:>8} {s:>12} {s:>12}\n", .{ "Size", "Time (us)", "GB/s" });
    print("-------------------------------------------\n", .{});

    const sizes = [_]usize{ 256, 512, 1024, 2048, 4096 };
    const allocator = std.heap.page_allocator;

    for (sizes) |n| {
        // Memory traffic: read A (n*n), read x (n), write y (n) = n*n + 2*n floats
        const bytes: f64 = (@as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(n)) + 2.0 * @as(f64, @floatFromInt(n))) * 4.0;

        const A = try allocator.alloc(f32, n * n);
        defer allocator.free(A);
        const x = try allocator.alloc(f32, n);
        defer allocator.free(x);
        const y = try allocator.alloc(f32, n);
        defer allocator.free(y);

        for (0..n * n) |i| {
            A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        }
        for (0..n) |i| {
            x[i] = @as(f32, @floatFromInt(i % 10)) / 10.0;
        }
        @memset(y, 0.0);

        // Warmup
        zblas.sgemv(n, n, A, x, y, 1.0, 0.0);

        // Benchmark
        const iterations: usize = if (n <= 1024) 100 else 20;
        var total_time: f64 = 0.0;

        for (0..iterations) |_| {
            @memset(y, 0.0);

            var timer = std.time.Timer.start() catch unreachable;
            zblas.sgemv(n, n, A, x, y, 1.0, 0.0);
            const elapsed = timer.read();

            total_time += @as(f64, @floatFromInt(elapsed)) / 1000.0; // ns to us
        }

        const avg_time = total_time / @as(f64, @floatFromInt(iterations));
        const bandwidth = bytes / (avg_time / 1_000_000.0) / 1e9; // GB/s

        print("{d:>8} {d:>12.2} {d:>12.2}\n", .{
            n,
            avg_time,
            bandwidth,
        });
    }

    print("\n", .{});
}
