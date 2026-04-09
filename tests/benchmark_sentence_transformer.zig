// zblas/tests/benchmark_sentence_transformer.zig
// Sentence Transformer workload benchmark for zblas
//
// Tests the exact SGEMM shapes used in all-MiniLM-L6-v2 inference:
//   - 6 BERT layers × (4 attention projections + 2 FFN layers) = 36 GEMMs
//   - All have small M (≈ sequence length, typically 8–30 tokens)
//   - This is the regime where zblas needs the most improvement
//
// Performance target: ≤5ms total for single sentence (from current ~12ms)
// PyTorch CPU baseline: ~3.6ms
//
// Run: zig build bench-st && ./zig-out/bin/zblas-benchmark-st

const std = @import("std");
const zblas = @import("zblas");

const print = std.debug.print;

pub fn main() !void {
    print("\n", .{});
    print("==============================================\n", .{});
    print("  Sentence Transformer SGEMM Benchmark        \n", .{});
    print("  Model: all-MiniLM-L6-v2 (384-dim, 6 layers) \n", .{});
    print("==============================================\n\n", .{});

    // Print system info
    const info = zblas.getInfo();
    print("Architecture: {s}\n", .{info.arch});
    print("SIMD enabled: {}\n", .{info.simd});
    print("Vector width: {} floats\n", .{info.vector_width});
    print("Micro-kernel: {}x{}\n\n", .{ info.mr, info.nr });

    const allocator = std.heap.page_allocator;

    // =========================================================================
    // Part 1: Per-shape benchmarks (individual GEMM timings)
    // =========================================================================
    print("Part 1: Individual GEMM Shapes\n", .{});
    print("----------------------------------------------\n", .{});
    print("{s:<40} {s:>10} {s:>10} {s:>10}\n", .{ "Operation", "M×N×K", "Time(µs)", "GFLOPS" });
    print("----------------------------------------------\n", .{});

    // Typical sentence: ~11 tokens after WordPiece tokenization
    const seq_len: usize = 11;
    const hidden: usize = 384;
    const intermediate: usize = 1536;
    const head_dim: usize = 32; // 384 / 12 heads
    const num_heads: usize = 12;

    const shapes = [_]struct { m: usize, n: usize, k: usize, name: []const u8, count: usize }{
        // Attention Q/K/V projections: seq_len × hidden → hidden (×3 per layer × 6 layers = 18)
        .{ .m = seq_len, .n = hidden, .k = hidden, .name = "Attn Q/K/V projection", .count = 18 },
        // Attention output projection: seq_len × hidden → hidden (×1 per layer × 6 = 6)
        .{ .m = seq_len, .n = hidden, .k = hidden, .name = "Attn output projection", .count = 6 },
        // FFN layer 1: seq_len × hidden → intermediate (×6 layers)
        .{ .m = seq_len, .n = intermediate, .k = hidden, .name = "FFN up (384→1536)", .count = 6 },
        // FFN layer 2: seq_len × intermediate → hidden (×6 layers)
        .{ .m = seq_len, .n = hidden, .k = intermediate, .name = "FFN down (1536→384)", .count = 6 },
        // Attention scores: seq × seq per head (×12 heads × 6 layers = 72)
        .{ .m = seq_len, .n = seq_len, .k = head_dim, .name = "Attn scores (per head)", .count = num_heads * 6 },
        // Attention weighted values: seq × head_dim per head (×12 heads × 6 layers = 72)
        .{ .m = seq_len, .n = head_dim, .k = seq_len, .name = "Attn weighted (per head)", .count = num_heads * 6 },
    };

    var total_single_us: f64 = 0.0;
    var total_inference_us: f64 = 0.0;
    var total_flops: f64 = 0.0;

    for (shapes) |shape| {
        const result = try benchmarkShape(allocator, shape.m, shape.n, shape.k);

        var dim_buf: [32]u8 = undefined;
        const dim_str = std.fmt.bufPrint(&dim_buf, "{d}×{d}×{d}", .{ shape.m, shape.n, shape.k }) catch "???";

        const inference_us = result.avg_us * @as(f64, @floatFromInt(shape.count));
        total_single_us += result.avg_us;
        total_inference_us += inference_us;

        const flops_per_call = 2.0 * @as(f64, @floatFromInt(shape.m)) *
            @as(f64, @floatFromInt(shape.n)) * @as(f64, @floatFromInt(shape.k));
        total_flops += flops_per_call * @as(f64, @floatFromInt(shape.count));

        print("{s:<40} {s:>10} {d:>10.1} {d:>10.1}\n", .{
            shape.name,
            dim_str,
            result.avg_us,
            result.gflops,
        });
    }

    print("----------------------------------------------\n\n", .{});

    // =========================================================================
    // Part 2: Full inference simulation (all 36+ GEMMs in sequence)
    // =========================================================================
    print("Part 2: Full Inference Simulation ({d} tokens)\n", .{seq_len});
    print("----------------------------------------------\n", .{});

    const full_result = try benchmarkFullInference(allocator, seq_len);

    print("  Total GEMMs per inference:  {d}\n", .{full_result.num_gemms});
    print("  Total FLOPS per inference:  {d:.1} MFLOPS\n", .{full_result.total_flops / 1e6});
    print("  Avg inference time:         {d:.2} ms\n", .{full_result.avg_ms});
    print("  Min inference time:         {d:.2} ms\n", .{full_result.min_ms});
    print("  Effective GFLOPS:           {d:.1}\n", .{full_result.gflops});
    print("\n", .{});

    // =========================================================================
    // Part 3: Vary sequence length to map performance curve
    // =========================================================================
    print("Part 3: Performance vs Sequence Length\n", .{});
    print("----------------------------------------------\n", .{});
    print("{s:>8} {s:>12} {s:>12} {s:>12}\n", .{ "SeqLen", "Time(ms)", "GFLOPS", "µs/token" });
    print("----------------------------------------------\n", .{});

    const seq_lengths = [_]usize{ 4, 8, 11, 16, 24, 32, 48, 64, 96, 128 };
    for (seq_lengths) |sl| {
        const r = try benchmarkFullInference(allocator, sl);
        print("{d:>8} {d:>12.2} {d:>12.1} {d:>12.1}\n", .{
            sl,
            r.avg_ms,
            r.gflops,
            r.avg_ms * 1000.0 / @as(f64, @floatFromInt(sl)),
        });
    }

    // =========================================================================
    // Part 4: Comparison targets
    // =========================================================================
    print("\n", .{});
    print("==============================================\n", .{});
    print("  Performance Targets                         \n", .{});
    print("==============================================\n", .{});
    print("  PyTorch CPU (MKL):     ~3.6 ms  (~64 GFLOPS eff)\n", .{});
    print("  PyTorch CUDA (4090):   ~2.6 ms\n", .{});
    print("  Theoretical peak:      ~56 GFLOPS (AVX2 FMA, single core)\n", .{});
    print("  zblas current:         {d:.2} ms  ({d:.1} GFLOPS eff)\n", .{
        full_result.avg_ms,
        full_result.gflops,
    });

    const speedup_needed = full_result.avg_ms / 3.6;
    print("  Gap vs PyTorch CPU:    {d:.1}×\n", .{speedup_needed});
    print("  Target:                ≤5 ms  (≥{d:.0} GFLOPS eff)\n", .{full_result.total_flops / 5e6});
    print("==============================================\n\n", .{});
}

const ShapeResult = struct {
    avg_us: f64,
    min_us: f64,
    gflops: f64,
};

fn benchmarkShape(allocator: std.mem.Allocator, M: usize, N: usize, K: usize) !ShapeResult {
    const A = try allocator.alloc(f32, M * K);
    defer allocator.free(A);
    const B = try allocator.alloc(f32, K * N);
    defer allocator.free(B);
    const C = try allocator.alloc(f32, M * N);
    defer allocator.free(C);

    // Initialize
    for (A, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 97)) / 97.0 - 0.5;
    for (B, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 101)) / 101.0 - 0.5;
    @memset(C, 0.0);

    // Warmup
    for (0..5) |_| {
        zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);
    }

    // Benchmark — more iterations for small GEMMs to get stable timings
    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    const iterations: usize = blk: {
        const target_ns: f64 = 200_000_000.0; // 200ms measurement window
        const est_ns = flops / 50e9 * 1e9;
        break :blk @min(@max(50, @as(usize, @intFromFloat(target_ns / @max(est_ns, 1.0)))), 10000);
    };

    var min_ns: u64 = std.math.maxInt(u64);
    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        const lap_start = timer.read();
        zblas.sgemm(M, N, K, A, B, C, 1.0, 0.0);
        const lap_end = timer.read();
        const lap = lap_end - lap_start;
        if (lap < min_ns) min_ns = lap;
    }

    const total_ns = timer.read();
    const avg_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations));
    const avg_us = avg_ns / 1000.0;
    const min_us = @as(f64, @floatFromInt(min_ns)) / 1000.0;
    const gflops = flops / avg_ns;

    _ = min_us;

    return .{
        .avg_us = avg_us,
        .min_us = @as(f64, @floatFromInt(min_ns)) / 1000.0,
        .gflops = gflops,
    };
}

const InferenceResult = struct {
    num_gemms: usize,
    total_flops: f64,
    avg_ms: f64,
    min_ms: f64,
    gflops: f64,
};

fn benchmarkFullInference(allocator: std.mem.Allocator, seq_len: usize) !InferenceResult {
    const hidden: usize = 384;
    const intermediate: usize = 1536;
    const head_dim: usize = 32;
    const num_heads: usize = 12;
    const num_layers: usize = 6;

    // Pre-allocate all buffers at maximum sizes needed
    const max_m: usize = seq_len;
    const max_n: usize = @max(hidden, intermediate);
    const max_k: usize = @max(hidden, intermediate);

    const A = try allocator.alloc(f32, @as(usize, max_m) * @as(usize, max_k));
    defer allocator.free(A);
    const B = try allocator.alloc(f32, @as(usize, max_k) * @as(usize, max_n));
    defer allocator.free(B);
    const C = try allocator.alloc(f32, @as(usize, max_m) * @as(usize, max_n));
    defer allocator.free(C);

    // Initialize once
    for (A, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 97)) / 97.0 - 0.5;
    for (B, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7) % 101)) / 101.0 - 0.5;

    var num_gemms: usize = 0;
    var total_flops: f64 = 0.0;

    // Count GEMMs and FLOPS
    for (0..num_layers) |_| {
        // 3 projection GEMMs (Q, K, V)
        num_gemms += 3;
        total_flops += 3.0 * 2.0 * @as(f64, @floatFromInt(seq_len * hidden * hidden));
        // Attention scores (per head)
        num_gemms += num_heads;
        total_flops += @as(f64, @floatFromInt(num_heads)) * 2.0 * @as(f64, @floatFromInt(seq_len * seq_len * head_dim));
        // Attention weighted (per head)
        num_gemms += num_heads;
        total_flops += @as(f64, @floatFromInt(num_heads)) * 2.0 * @as(f64, @floatFromInt(seq_len * head_dim * seq_len));
        // Output projection
        num_gemms += 1;
        total_flops += 2.0 * @as(f64, @floatFromInt(seq_len * hidden * hidden));
        // FFN layer 1
        num_gemms += 1;
        total_flops += 2.0 * @as(f64, @floatFromInt(seq_len * intermediate * hidden));
        // FFN layer 2
        num_gemms += 1;
        total_flops += 2.0 * @as(f64, @floatFromInt(seq_len * hidden * intermediate));
    }

    // Warmup: run one full inference
    runInference(seq_len, hidden, intermediate, head_dim, num_heads, num_layers, A, B, C);

    // Benchmark
    const iterations: usize = 20;
    var min_ns: u64 = std.math.maxInt(u64);
    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        const lap_start = timer.read();
        runInference(seq_len, hidden, intermediate, head_dim, num_heads, num_layers, A, B, C);
        const lap_end = timer.read();
        const lap = lap_end - lap_start;
        if (lap < min_ns) min_ns = lap;
    }

    const total_ns = timer.read();
    const avg_ns = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations));
    const avg_ms = avg_ns / 1_000_000.0;
    const min_ms = @as(f64, @floatFromInt(min_ns)) / 1_000_000.0;
    const gflops = total_flops / avg_ns;

    return .{
        .num_gemms = num_gemms,
        .total_flops = total_flops,
        .avg_ms = avg_ms,
        .min_ms = min_ms,
        .gflops = gflops,
    };
}

fn runInference(
    seq_len: usize,
    hidden: usize,
    intermediate: usize,
    head_dim: usize,
    num_heads: usize,
    num_layers: usize,
    A: []f32,
    B: []f32,
    C: []f32,
) void {
    for (0..num_layers) |_| {
        // Q, K, V projections
        for (0..3) |_| {
            zblas.sgemm(seq_len, hidden, hidden, A, B, C, 1.0, 0.0);
        }
        // Attention scores per head
        for (0..num_heads) |_| {
            zblas.sgemm(seq_len, seq_len, head_dim, A, B, C, 1.0, 0.0);
        }
        // Attention weighted per head
        for (0..num_heads) |_| {
            zblas.sgemm(seq_len, head_dim, seq_len, A, B, C, 1.0, 0.0);
        }
        // Output projection
        zblas.sgemm(seq_len, hidden, hidden, A, B, C, 1.0, 0.0);
        // FFN up
        zblas.sgemm(seq_len, intermediate, hidden, A, B, C, 1.0, 0.0);
        // FFN down
        zblas.sgemm(seq_len, hidden, intermediate, A, B, C, 1.0, 0.0);
    }
}
