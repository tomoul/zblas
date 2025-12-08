// zblas/tests/benchmark_whisper.zig
// Whisper-specific benchmarks for Agent 9 Direct SIMD optimization
//
// These matrix dimensions are representative of Whisper-tiny inference workloads.
// The goal is to match or beat the ops.zig fallback performance (~4.6s total).

const std = @import("std");
const zblas = @import("zblas");

const print = std.debug.print;

pub fn main() !void {
    print("\n", .{});
    print("===========================================\n", .{});
    print("     Whisper Matrix Dimensions Benchmark   \n", .{});
    print("===========================================\n\n", .{});

    // Print system info
    const info = zblas.getInfo();
    print("Architecture: {s}\n", .{info.arch});
    print("SIMD enabled: {}\n", .{info.simd});
    print("Vector width: {} floats\n", .{info.vector_width});
    print("Micro-kernel: {}x{}\n", .{ info.mr, info.nr });
    print("\n", .{});

    const allocator = std.heap.page_allocator;

    // Whisper matrix dimensions from Agent 9 spec
    // These are the critical sizes for Whisper-tiny inference
    const whisper_sizes = [_]struct { m: usize, n: usize, k: usize, name: []const u8 }{
        .{ .m = 80, .n = 3000, .k = 512, .name = "Encoder attention (skinny-tall)" },
        .{ .m = 384, .n = 512, .k = 512, .name = "FFN layer 1" },
        .{ .m = 512, .n = 384, .k = 512, .name = "FFN layer 2" },
        .{ .m = 64, .n = 64, .k = 512, .name = "Decoder attention (small)" },
        .{ .m = 1500, .n = 512, .k = 512, .name = "Encoder output (tall)" },
        // Additional sizes for edge case testing
        .{ .m = 256, .n = 256, .k = 256, .name = "Medium square" },
        .{ .m = 512, .n = 512, .k = 512, .name = "Large square" },
        .{ .m = 1024, .n = 1024, .k = 1024, .name = "Larger square" },
    };

    print("Individual Matrix Benchmarks\n", .{});
    print("-------------------------------------------\n", .{});
    print("{s:<36} {s:>10} {s:>10} {s:>10}\n", .{ "Operation", "M×N×K", "Time(ms)", "GFLOPS" });
    print("-------------------------------------------\n", .{});

    var total_time_ms: f64 = 0.0;
    var total_flops: f64 = 0.0;

    for (whisper_sizes) |dims| {
        const M = dims.m;
        const N = dims.n;
        const K = dims.k;

        // Allocate matrices
        const A = try allocator.alloc(f32, M * K);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, K * N);
        defer allocator.free(B);
        const C = try allocator.alloc(f32, M * N);
        defer allocator.free(C);

        // Initialize with reproducible values
        for (A, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
        for (B, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
        @memset(C, 0.0);

        // Warmup
        zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);

        // Benchmark - more iterations for small matrices
        const flops_per_call = 2.0 * @as(f64, @floatFromInt(M)) *
            @as(f64, @floatFromInt(N)) *
            @as(f64, @floatFromInt(K));
        const iterations: usize = blk: {
            // Target ~100ms minimum measurement time
            const target_ns: f64 = 100_000_000.0;
            const est_ns_per_call = flops_per_call / 50e9 * 1e9; // Assume ~50 GFLOPS
            const iters = @max(10, @as(usize, @intFromFloat(target_ns / est_ns_per_call)));
            break :blk @min(iters, 1000);
        };

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const avg_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));
        const total_flops_run = flops_per_call * @as(f64, @floatFromInt(iterations));
        const gflops = total_flops_run / @as(f64, @floatFromInt(elapsed_ns));

        total_time_ms += avg_ms;
        total_flops += flops_per_call;

        // Format dimension string
        var dim_buf: [32]u8 = undefined;
        const dim_str = std.fmt.bufPrint(&dim_buf, "{d}×{d}×{d}", .{ M, N, K }) catch "???";

        print("{s:<36} {s:>10} {d:>10.3} {d:>10.1}\n", .{
            dims.name,
            dim_str,
            avg_ms,
            gflops,
        });
    }

    print("-------------------------------------------\n", .{});
    print("{s:<36} {s:>10} {d:>10.3} {d:>10.1}\n", .{
        "TOTAL (sum of averages)",
        "",
        total_time_ms,
        total_flops / (total_time_ms / 1000.0) / 1e9,
    });
    print("\n", .{});

    // Shape-specific analysis
    print("Shape Analysis (grouped by matrix type)\n", .{});
    print("-------------------------------------------\n", .{});
    try benchmarkByShape(allocator);

    print("\n===========================================\n", .{});
    print("           Benchmark Complete              \n", .{});
    print("===========================================\n", .{});
}

fn benchmarkByShape(allocator: std.mem.Allocator) !void {
    // Test skinny matrices (M << N or N << M)
    const skinny_sizes = [_][3]usize{
        .{ 64, 1024, 512 }, // Skinny-tall
        .{ 1024, 64, 512 }, // Tall-skinny
        .{ 32, 2048, 256 }, // Very skinny
        .{ 2048, 32, 256 }, // Very tall
    };

    print("\nSkinny Matrices (M << N or N << M):\n", .{});
    for (skinny_sizes) |dims| {
        const M, const N, const K = dims;
        try benchmarkSingle(allocator, M, N, K);
    }

    // Test square-ish matrices
    const square_sizes = [_][3]usize{
        .{ 128, 128, 128 },
        .{ 256, 256, 256 },
        .{ 512, 512, 512 },
        .{ 768, 768, 768 },
    };

    print("\nSquare-ish Matrices:\n", .{});
    for (square_sizes) |dims| {
        const M, const N, const K = dims;
        try benchmarkSingle(allocator, M, N, K);
    }

    // Test matrices near the 2048 threshold
    const threshold_sizes = [_][3]usize{
        .{ 1024, 1024, 1024 },
        .{ 1536, 1536, 1536 },
        .{ 2000, 2000, 2000 },
        .{ 2048, 2048, 2048 }, // At threshold
        .{ 2049, 2049, 2049 }, // Just over threshold (blocked path)
    };

    print("\nNear 2048 Threshold (Direct vs Blocked):\n", .{});
    for (threshold_sizes) |dims| {
        const M, const N, const K = dims;
        try benchmarkSingle(allocator, M, N, K);
    }
}

fn benchmarkSingle(allocator: std.mem.Allocator, M: usize, N: usize, K: usize) !void {
    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, K * N);
    defer allocator.free(B);
    const C = try allocator.alloc(f32, M * N);
    defer allocator.free(C);

    // Initialize
    for (A, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    for (B, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
    @memset(C, 0.0);

    // Warmup
    zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);

    // Benchmark
    const flops = 2.0 * @as(f64, @floatFromInt(M * N * K));
    const iterations: usize = @max(3, @min(100, @as(usize, @intFromFloat(1e10 / flops))));

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const avg_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const gflops = (flops * @as(f64, @floatFromInt(iterations))) / @as(f64, @floatFromInt(elapsed_ns));

    print("  {d:>5}×{d:<5}×{d:<5}: {d:>8.3}ms, {d:>6.1} GFLOPS\n", .{
        M,
        N,
        K,
        avg_ms,
        gflops,
    });
}
