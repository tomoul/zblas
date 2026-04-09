// zblas/tests/benchmark_cache.zig
// Cache blocking parameter benchmark for
//
// This benchmark tests different KC (K-dimension blocking) values to find
// optimal cache utilization. The key insight from OpenBLAS analysis:
//   - OpenBLAS uses KC=256, MC=320, NR=4
//   - zblas uses KC=512, MC=256, NR=8
//   - Smaller KC keeps packed A in L2 cache

const std = @import("std");
const builtin = @import("builtin");

const print = std.debug.print;

pub fn main() !void {
    print("\n", .{});
    print("===========================================\n", .{});
    print("    Cache Blocking Parameter Benchmark     \n", .{});
    print("===========================================\n\n", .{});

    // Print cache size info
    print("Cache Size Analysis:\n", .{});
    print("-------------------------------------------\n", .{});
    print("Typical L1 data cache: 32 KB\n", .{});
    print("Typical L2 cache: 256 KB\n", .{});
    print("Typical L3 cache: 8-32 MB (shared)\n", .{});
    print("\n", .{});

    // Test different KC values
    const kc_values = [_]usize{ 128, 192, 256, 384, 512 };
    const mc_values = [_]usize{ 256, 320 };

    print("Packed A sizes (MC x KC x 4 bytes):\n", .{});
    print("-------------------------------------------\n", .{});
    for (mc_values) |mc| {
        for (kc_values) |kc| {
            const packed_a_kb = mc * kc * 4 / 1024;
            const fits_l2 = packed_a_kb <= 256;
            print("  MC={d:3}, KC={d:3}: {d:4} KB {s}\n", .{
                mc,
                kc,
                packed_a_kb,
                if (fits_l2) "(fits L2)" else "(spills to L3)",
            });
        }
    }
    print("\n", .{});

    // Benchmark with different configurations
    try benchmarkKcValues();

    // Test Whisper-like matrix sizes
    try benchmarkWhisperSizes();
}

/// Run SGEMM with a specific KC value by manually implementing the blocked algorithm
fn sgemmWithKc(
    comptime mc: usize,
    comptime kc: usize,
    comptime nc: usize,
    comptime mr: usize,
    comptime nr: usize,
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
) void {
    // Allocate packing buffers on stack (be careful with sizes)
    var packed_a: [mc * kc]f32 = undefined;
    var packed_b: [kc * nc]f32 = undefined;

    // Scale C by beta (here beta=0)
    for (0..M) |i| {
        @memset(C[i * N ..][0..N], 0.0);
    }

    // Main blocking loops (GotoBLAS algorithm)
    var jc: usize = 0;
    while (jc < N) : (jc += nc) {
        const jb = @min(nc, N - jc);

        var pc: usize = 0;
        while (pc < K) : (pc += kc) {
            const pb = @min(kc, K - pc);

            // Pack B panel [pb x jb]
            packB(B[pc * N + jc ..], N, &packed_b, pb, jb, nr);

            var ic: usize = 0;
            while (ic < M) : (ic += mc) {
                const ib = @min(mc, M - ic);

                // Pack A block [ib x pb]
                packA(A[ic * K + pc ..], K, &packed_a, ib, pb, mr);

                // Compute C[ic:ic+ib, jc:jc+jb] += packed_A * packed_B
                computeBlockGeneric(mr, nr, &packed_a, &packed_b, C[ic * N + jc ..], N, ib, jb, pb);
            }
        }
    }
}

fn packA(
    src: []const f32,
    lda: usize,
    dest: []f32,
    m: usize,
    k: usize,
    mr: usize,
) void {
    var dest_idx: usize = 0;
    var i: usize = 0;

    // Full MR panels
    while (i + mr <= m) : (i += mr) {
        for (0..k) |kk| {
            for (0..mr) |ii| {
                dest[dest_idx] = src[(i + ii) * lda + kk];
                dest_idx += 1;
            }
        }
    }

    // Remainder rows (< MR)
    if (i < m) {
        const remaining = m - i;
        for (0..k) |kk| {
            for (0..mr) |ii| {
                if (ii < remaining) {
                    dest[dest_idx] = src[(i + ii) * lda + kk];
                } else {
                    dest[dest_idx] = 0.0;
                }
                dest_idx += 1;
            }
        }
    }
}

fn packB(
    src: []const f32,
    ldb: usize,
    dest: []f32,
    k: usize,
    n: usize,
    nr: usize,
) void {
    var dest_idx: usize = 0;
    var j: usize = 0;

    // Full NR panels
    while (j + nr <= n) : (j += nr) {
        for (0..k) |kk| {
            for (0..nr) |jj| {
                dest[dest_idx] = src[kk * ldb + j + jj];
                dest_idx += 1;
            }
        }
    }

    // Remainder columns (< NR)
    if (j < n) {
        const remaining = n - j;
        for (0..k) |kk| {
            for (0..nr) |jj| {
                if (jj < remaining) {
                    dest[dest_idx] = src[kk * ldb + j + jj];
                } else {
                    dest[dest_idx] = 0.0;
                }
                dest_idx += 1;
            }
        }
    }
}

fn computeBlockGeneric(
    comptime mr: usize,
    comptime nr: usize,
    packed_a: []const f32,
    packed_b: []const f32,
    C: []f32,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) void {
    // Process NR×MR micro-tiles
    var jr: usize = 0;
    while (jr < n) : (jr += nr) {
        const nr_actual = @min(nr, n - jr);

        var ir: usize = 0;
        while (ir < m) : (ir += mr) {
            const mr_actual = @min(mr, m - ir);

            // Scalar micro-kernel for simplicity
            for (0..mr_actual) |i| {
                for (0..nr_actual) |j| {
                    var sum: f32 = 0.0;
                    for (0..k) |kk| {
                        sum += packed_a[ir * k + kk * mr + i] * packed_b[jr * k + kk * nr + j];
                    }
                    C[(ir + i) * ldc + jr + j] += sum;
                }
            }
        }
    }
}

fn benchmarkKcValues() !void {
    print("KC Value Benchmark (512x512 matrix)\n", .{});
    print("-------------------------------------------\n", .{});
    print("{s:>6} {s:>6} {s:>6} {s:>12} {s:>12}\n", .{ "MC", "KC", "NR", "Time (ms)", "GFLOPS" });
    print("-------------------------------------------\n", .{});

    const n: usize = 512;
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(n)) *
        @as(f64, @floatFromInt(n)) *
        @as(f64, @floatFromInt(n));

    const allocator = std.heap.page_allocator;

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

    // Test configurations
    const configs = [_]struct { mc: usize, kc: usize, nr: usize }{
        .{ .mc = 256, .kc = 128, .nr = 8 },
        .{ .mc = 256, .kc = 192, .nr = 8 },
        .{ .mc = 256, .kc = 256, .nr = 8 },
        .{ .mc = 256, .kc = 384, .nr = 8 },
        .{ .mc = 256, .kc = 512, .nr = 8 }, // Current zblas
        .{ .mc = 320, .kc = 256, .nr = 4 }, // OpenBLAS style
        .{ .mc = 320, .kc = 256, .nr = 8 },
    };

    for (configs) |cfg| {
        @memset(C, 0.0);

        // Warmup - use comptime-known sizes
        switch (cfg.kc) {
            128 => {
                if (cfg.mc == 256 and cfg.nr == 8) {
                    sgemmWithKc(256, 128, 4096, 8, 8, n, n, n, A, B, C);
                }
            },
            192 => {
                if (cfg.mc == 256 and cfg.nr == 8) {
                    sgemmWithKc(256, 192, 4096, 8, 8, n, n, n, A, B, C);
                }
            },
            256 => {
                if (cfg.mc == 256 and cfg.nr == 8) {
                    sgemmWithKc(256, 256, 4096, 8, 8, n, n, n, A, B, C);
                } else if (cfg.mc == 320 and cfg.nr == 4) {
                    sgemmWithKc(320, 256, 4096, 8, 4, n, n, n, A, B, C);
                } else if (cfg.mc == 320 and cfg.nr == 8) {
                    sgemmWithKc(320, 256, 4096, 8, 8, n, n, n, A, B, C);
                }
            },
            384 => {
                if (cfg.mc == 256 and cfg.nr == 8) {
                    sgemmWithKc(256, 384, 4096, 8, 8, n, n, n, A, B, C);
                }
            },
            512 => {
                if (cfg.mc == 256 and cfg.nr == 8) {
                    sgemmWithKc(256, 512, 4096, 8, 8, n, n, n, A, B, C);
                }
            },
            else => continue,
        }

        // Benchmark
        const iterations: usize = 5;
        var total_time: f64 = 0.0;

        for (0..iterations) |_| {
            @memset(C, 0.0);

            var timer = std.time.Timer.start() catch unreachable;

            switch (cfg.kc) {
                128 => {
                    if (cfg.mc == 256 and cfg.nr == 8) {
                        sgemmWithKc(256, 128, 4096, 8, 8, n, n, n, A, B, C);
                    }
                },
                192 => {
                    if (cfg.mc == 256 and cfg.nr == 8) {
                        sgemmWithKc(256, 192, 4096, 8, 8, n, n, n, A, B, C);
                    }
                },
                256 => {
                    if (cfg.mc == 256 and cfg.nr == 8) {
                        sgemmWithKc(256, 256, 4096, 8, 8, n, n, n, A, B, C);
                    } else if (cfg.mc == 320 and cfg.nr == 4) {
                        sgemmWithKc(320, 256, 4096, 8, 4, n, n, n, A, B, C);
                    } else if (cfg.mc == 320 and cfg.nr == 8) {
                        sgemmWithKc(320, 256, 4096, 8, 8, n, n, n, A, B, C);
                    }
                },
                384 => {
                    if (cfg.mc == 256 and cfg.nr == 8) {
                        sgemmWithKc(256, 384, 4096, 8, 8, n, n, n, A, B, C);
                    }
                },
                512 => {
                    if (cfg.mc == 256 and cfg.nr == 8) {
                        sgemmWithKc(256, 512, 4096, 8, 8, n, n, n, A, B, C);
                    }
                },
                else => continue,
            }

            const elapsed = timer.read();
            total_time += @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
        }

        const avg_time = total_time / @as(f64, @floatFromInt(iterations));
        const gflops = flops / (avg_time / 1000.0) / 1e9;

        print("{d:>6} {d:>6} {d:>6} {d:>12.3} {d:>12.2}\n", .{
            cfg.mc,
            cfg.kc,
            cfg.nr,
            avg_time,
            gflops,
        });
    }

    print("\n", .{});
}

fn benchmarkWhisperSizes() !void {
    print("Whisper-like Matrix Size Benchmark\n", .{});
    print("-------------------------------------------\n", .{});
    print("These are common matrix sizes in Whisper inference:\n", .{});
    print("  - Encoder attention: ~80x3000x512\n", .{});
    print("  - Feed-forward: ~384x512x512\n", .{});
    print("-------------------------------------------\n", .{});

    const allocator = std.heap.page_allocator;

    // Import zblas for comparison
    const zblas = @import("zblas");

    const test_sizes = [_]struct { m: usize, n: usize, k: usize, name: []const u8 }{
        .{ .m = 80, .n = 3000, .k = 512, .name = "Encoder attention" },
        .{ .m = 384, .n = 512, .k = 512, .name = "Feed-forward 1" },
        .{ .m = 512, .n = 384, .k = 512, .name = "Feed-forward 2" },
        .{ .m = 512, .n = 512, .k = 512, .name = "Standard GEMM" },
    };

    print("\n{s:<20} {s:>10} {s:>10} {s:>10} {s:>12}\n", .{ "Workload", "M", "N", "K", "GFLOPS" });
    print("-------------------------------------------\n", .{});

    for (test_sizes) |size| {
        const m = size.m;
        const n = size.n;
        const k = size.k;
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) *
            @as(f64, @floatFromInt(n)) *
            @as(f64, @floatFromInt(k));

        const A = try allocator.alloc(f32, m * k);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, k * n);
        defer allocator.free(B);
        const C = try allocator.alloc(f32, m * n);
        defer allocator.free(C);

        for (0..m * k) |i| {
            A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        }
        for (0..k * n) |i| {
            B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
        }
        @memset(C, 0.0);

        // Warmup
        zblas.sgemm(m, n, k, A, B, C, 1.0, 0.0);

        // Benchmark
        const iterations: usize = 10;
        var total_time: f64 = 0.0;

        for (0..iterations) |_| {
            @memset(C, 0.0);

            var timer = std.time.Timer.start() catch unreachable;
            zblas.sgemm(m, n, k, A, B, C, 1.0, 0.0);
            const elapsed = timer.read();

            total_time += @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
        }

        const avg_time = total_time / @as(f64, @floatFromInt(iterations));
        const gflops = flops / (avg_time / 1000.0) / 1e9;

        print("{s:<20} {d:>10} {d:>10} {d:>10} {d:>12.2}\n", .{
            size.name,
            m,
            n,
            k,
            gflops,
        });
    }

    print("\n", .{});

    // Test large matrices (triggers cache-blocked path)
    try benchmarkLargeMatrices();
}

fn benchmarkLargeMatrices() !void {
    print("Large Matrix Benchmark (Cache-blocked path)\n", .{});
    print("-------------------------------------------\n", .{});
    print("Matrices > 2048 use cache-blocked GEMM with packing.\n", .{});
    print("This tests the impact of KC, MC cache parameters.\n", .{});
    print("-------------------------------------------\n", .{});

    const allocator = std.heap.page_allocator;
    const zblas = @import("zblas");

    // Test sizes that trigger the cache-blocked path
    const large_sizes = [_]usize{ 2048, 2560, 3072 };

    print("\n{s:>8} {s:>12} {s:>12} {s:>12}\n", .{ "Size", "Time (ms)", "GFLOPS", "Path" });
    print("-------------------------------------------\n", .{});

    for (large_sizes) |n| {
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(n)) *
            @as(f64, @floatFromInt(n)) *
            @as(f64, @floatFromInt(n));

        const A = try allocator.alloc(f32, n * n);
        defer allocator.free(A);
        const B = try allocator.alloc(f32, n * n);
        defer allocator.free(B);
        const C = try allocator.alloc(f32, n * n);
        defer allocator.free(C);

        for (0..n * n) |i| {
            A[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
            B[i] = @as(f32, @floatFromInt((i * 7) % 100)) / 100.0;
        }
        @memset(C, 0.0);

        // Warmup
        zblas.sgemm(n, n, n, A, B, C, 1.0, 0.0);

        // Benchmark
        const iterations: usize = 3;
        var total_time: f64 = 0.0;

        for (0..iterations) |_| {
            @memset(C, 0.0);

            var timer = std.time.Timer.start() catch unreachable;
            zblas.sgemm(n, n, n, A, B, C, 1.0, 0.0);
            const elapsed = timer.read();

            total_time += @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
        }

        const avg_time = total_time / @as(f64, @floatFromInt(iterations));
        const gflops = flops / (avg_time / 1000.0) / 1e9;
        const path = if (n > 2048) "blocked" else "direct";

        print("{d:>8} {d:>12.2} {d:>12.2} {s:>12}\n", .{
            n,
            avg_time,
            gflops,
            path,
        });
    }

    print("\n", .{});
    print("Cache config analysis:\n", .{});
    print("  MC={}, KC={}, NC={}\n", .{ zblas.MC, zblas.KC, zblas.NC });
    print("  Packed A size: {} KB\n", .{zblas.MC * zblas.KC * 4 / 1024});
    print("  Packed B size: {} KB\n", .{zblas.KC * zblas.NC * 4 / 1024});
    print("\n", .{});
}
