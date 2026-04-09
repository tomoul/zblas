// zblas/tests/benchmark_q8_gemm.zig
// Q8 SGEMM benchmarks: naive vs zblas for XLM-RoBERTa workload sizes

const std = @import("std");
const zblas = @import("zblas");

const print = std.debug.print;

pub fn main() !void {
    print("\n", .{});
    print("===========================================\n", .{});
    print("     Q8 Weight-Only SGEMM Benchmarks      \n", .{});
    print("===========================================\n\n", .{});

    const info = zblas.getInfo();
    print("Architecture: {s}\n", .{info.arch});
    print("SIMD enabled: {}\n", .{info.simd});
    print("Vector width: {} floats\n", .{info.vector_width});
    print("Micro-kernel: {}x{}\n", .{ info.mr, info.nr });
    print("CPU cores: {}\n\n", .{std.Thread.getCpuCount() catch 1});

    // XLM-RoBERTa Large workload sizes
    // seq_len=9 (short), seq_len=64 (medium), seq_len=128 (long)
    const workloads = [_]struct { name: []const u8, M: usize, N: usize, K: usize }{
        // Short input (9 tokens) — attention QKV/output projections
        .{ .name = "Attn QKV (9×1024×1024)", .M = 9, .N = 1024, .K = 1024 },
        // Short input — FFN up projection
        .{ .name = "FFN up  (9×4096×1024)", .M = 9, .N = 4096, .K = 1024 },
        // Short input — FFN down projection
        .{ .name = "FFN down(9×1024×4096)", .M = 9, .N = 1024, .K = 4096 },

        // Medium input (64 tokens)
        .{ .name = "Attn QKV(64×1024×1024)", .M = 64, .N = 1024, .K = 1024 },
        .{ .name = "FFN up (64×4096×1024)", .M = 64, .N = 4096, .K = 1024 },
        .{ .name = "FFN down(64×1024×4096)", .M = 64, .N = 1024, .K = 4096 },

        // Large input (128 tokens)
        .{ .name = "Attn QKV(128×1024×1024)", .M = 128, .N = 1024, .K = 1024 },
        .{ .name = "FFN up (128×4096×1024)", .M = 128, .N = 4096, .K = 1024 },
        .{ .name = "FFN down(128×1024×4096)", .M = 128, .N = 1024, .K = 4096 },

        // Square (for comparison)
        .{ .name = "Square  (256×256×256)", .M = 256, .N = 256, .K = 256 },
        .{ .name = "Square  (512×512×512)", .M = 512, .N = 512, .K = 512 },
        .{ .name = "Square (1024×1024×1024)", .M = 1024, .N = 1024, .K = 1024 },
    };

    const allocator = std.heap.page_allocator;

    print("Naive Q8 Matmul (i,k,j loop with SIMD)\n", .{});
    print("-------------------------------------------------------------\n", .{});
    print("{s:>28} {s:>10} {s:>10} {s:>10}\n", .{ "Workload", "Time(ms)", "GFLOPS", "Iters" });
    print("-------------------------------------------------------------\n", .{});

    // Collect naive times for speedup comparison
    var naive_times: [workloads.len]f64 = undefined;

    for (workloads, 0..) |wl, wi| {
        const result = try benchmarkNaiveQ8(allocator, wl.M, wl.N, wl.K);
        naive_times[wi] = result.time_ms;
        print("{s:>28} {d:>10.3} {d:>10.2} {d:>10}\n", .{ wl.name, result.time_ms, result.gflops, result.iters });
    }

    print("\nzblas sgemmQ8 (cache-blocked dequant)\n", .{});
    print("-------------------------------------------------------------\n", .{});
    print("{s:>28} {s:>10} {s:>10} {s:>10} {s:>10}\n", .{ "Workload", "Time(ms)", "GFLOPS", "Iters", "Speedup" });
    print("-------------------------------------------------------------\n", .{});

    for (workloads, 0..) |wl, wi| {
        const result = try benchmarkZblasQ8(allocator, wl.M, wl.N, wl.K);
        const speedup = naive_times[wi] / result.time_ms;
        print("{s:>28} {d:>10.3} {d:>10.2} {d:>10} {d:>9.2}×\n", .{ wl.name, result.time_ms, result.gflops, result.iters, speedup });
    }

    print("\n===========================================\n", .{});
    print("           Benchmark Complete              \n", .{});
    print("===========================================\n", .{});
}

const BenchResult = struct {
    time_ms: f64,
    gflops: f64,
    iters: usize,
};

fn benchmarkNaiveQ8(allocator: std.mem.Allocator, M: usize, N: usize, K: usize) !BenchResult {
    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B_q8 = try allocator.alloc(i8, K * N);
    defer allocator.free(B_q8);
    const C = try allocator.alloc(f32, M * N);
    defer allocator.free(C);

    // Initialize
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B_q8) |*v| v.* = @as(i8, @intCast(@as(i32, random.intRangeAtMost(u8, 0, 255)) - 128));
    @memset(C, 0.0);

    const scale: f32 = 0.00784313725; // 1.0 / 127.0 roughly

    // Warmup
    naiveQ8Matmul(M, N, K, A, B_q8, scale, C);

    // Determine iterations based on problem size
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const iters: usize = @max(3, @min(100, @as(usize, @intFromFloat(1e9 / flops))));

    var timer = try std.time.Timer.start();
    for (0..iters) |_| {
        @memset(C, 0.0);
        naiveQ8Matmul(M, N, K, A, B_q8, scale, C);
    }
    const elapsed_ns = timer.read();
    const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1e6 / @as(f64, @floatFromInt(iters));
    const gflops = flops / (time_ms * 1e6);

    return .{ .time_ms = time_ms, .gflops = gflops, .iters = iters };
}

fn benchmarkZblasQ8(allocator: std.mem.Allocator, M: usize, N: usize, K: usize) !BenchResult {
    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B_q8 = try allocator.alloc(i8, K * N);
    defer allocator.free(B_q8);
    const C = try allocator.alloc(f32, M * N);
    defer allocator.free(C);

    // Initialize (same seed for consistency)
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (A) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (B_q8) |*v| v.* = @as(i8, @intCast(@as(i32, random.intRangeAtMost(u8, 0, 255)) - 128));
    @memset(C, 0.0);

    const scale: f32 = 0.00784313725;

    // Warmup
    zblas.sgemmQ8(M, N, K, A, B_q8, scale, C);

    const flops: f64 = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const iters: usize = @max(3, @min(100, @as(usize, @intFromFloat(1e9 / flops))));

    var timer = try std.time.Timer.start();
    for (0..iters) |_| {
        @memset(C, 0.0);
        zblas.sgemmQ8(M, N, K, A, B_q8, scale, C);
    }
    const elapsed_ns = timer.read();
    const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1e6 / @as(f64, @floatFromInt(iters));
    const gflops = flops / (time_ms * 1e6);

    return .{ .time_ms = time_ms, .gflops = gflops, .iters = iters };
}

/// Naive Q8 matmul matching Tomoul's matmulF32Q8Simd pattern:
/// C[i][j] += A[i][k] * (B_q8[k][j] * scale)
fn naiveQ8Matmul(M: usize, N: usize, K: usize, A: []const f32, B_q8: []const i8, scale: f32, C: []f32) void {
    const VEC_SIZE = 8; // AVX2 width

    for (0..M) |i| {
        for (0..K) |k| {
            const a_val = A[i * K + k];
            const a_vec: @Vector(VEC_SIZE, f32) = @splat(a_val);
            const scale_vec: @Vector(VEC_SIZE, f32) = @splat(scale);

            var j: usize = 0;
            while (j + VEC_SIZE <= N) : (j += VEC_SIZE) {
                // Load int8 → int32 → float32
                var b_i8: [VEC_SIZE]i8 = undefined;
                for (0..VEC_SIZE) |v| {
                    b_i8[v] = B_q8[k * N + j + v];
                }
                const b_i8_vec: @Vector(VEC_SIZE, i8) = b_i8;
                const b_i32: @Vector(VEC_SIZE, i32) = @intCast(b_i8_vec);
                const b_f32: @Vector(VEC_SIZE, f32) = @floatFromInt(b_i32);
                const dequant = b_f32 * scale_vec;

                // Load C, FMA, store
                const c_ptr = C[i * N + j ..][0..VEC_SIZE];
                const c_vec: @Vector(VEC_SIZE, f32) = c_ptr.*;
                c_ptr.* = @mulAdd(@Vector(VEC_SIZE, f32), a_vec, dequant, c_vec);

                j += 0; // loop increment handled by while
            }

            // Scalar tail
            while (j < N) : (j += 1) {
                C[i * N + j] += a_val * @as(f32, @floatFromInt(B_q8[k * N + j])) * scale;
            }
        }
    }
}
