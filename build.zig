const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ==========================================================================
    // zblas Library Module
    // ==========================================================================
    const zblas_module = b.addModule("zblas", .{
        .root_source_file = b.path("src/zblas.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ==========================================================================
    // Static Library
    // ==========================================================================
    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/zblas.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "zblas",
        .root_module = lib_module,
    });

    b.installArtifact(lib);

    // ==========================================================================
    // Unit Tests
    // ==========================================================================
    const test_module = b.createModule(.{
        .root_source_file = b.path("tests/test_sgemm.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("zblas", zblas_module);

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run zblas unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ==========================================================================
    // Benchmarks
    // ==========================================================================
    const bench_module = b.createModule(.{
        .root_source_file = b.path("tests/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast, // Always optimize benchmarks
    });
    bench_module.addImport("zblas", zblas_module);

    const benchmark = b.addExecutable(.{
        .name = "zblas-benchmark",
        .root_module = bench_module,
    });

    b.installArtifact(benchmark);

    const run_benchmark = b.addRunArtifact(benchmark);
    const bench_step = b.step("bench", "Run zblas benchmarks");
    bench_step.dependOn(&run_benchmark.step);

    // ==========================================================================
    // Whisper-specific Benchmark
    // ==========================================================================
    const whisper_bench_module = b.createModule(.{
        .root_source_file = b.path("tests/benchmark_whisper.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    whisper_bench_module.addImport("zblas", zblas_module);

    const whisper_benchmark = b.addExecutable(.{
        .name = "zblas-whisper-benchmark",
        .root_module = whisper_bench_module,
    });

    b.installArtifact(whisper_benchmark);

    const run_whisper_benchmark = b.addRunArtifact(whisper_benchmark);
    const whisper_bench_step = b.step("bench-whisper", "Run Whisper-specific benchmarks");
    whisper_bench_step.dependOn(&run_whisper_benchmark.step);

    // ==========================================================================
    // Cache Blocking Benchmark
    // ==========================================================================
    const cache_bench_module = b.createModule(.{
        .root_source_file = b.path("tests/benchmark_cache.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    cache_bench_module.addImport("zblas", zblas_module);

    const cache_benchmark = b.addExecutable(.{
        .name = "zblas-cache-benchmark",
        .root_module = cache_bench_module,
    });

    b.installArtifact(cache_benchmark);

    const run_cache_benchmark = b.addRunArtifact(cache_benchmark);
    const cache_bench_step = b.step("bench-cache", "Run cache blocking benchmarks");
    cache_bench_step.dependOn(&run_cache_benchmark.step);

    // ==========================================================================
    // Reference Tests (verify against naive implementation)
    // ==========================================================================
    const ref_test_module = b.createModule(.{
        .root_source_file = b.path("tests/test_reference.zig"),
        .target = target,
        .optimize = optimize,
    });
    ref_test_module.addImport("zblas", zblas_module);

    const ref_tests = b.addTest(.{
        .root_module = ref_test_module,
    });

    const run_ref_tests = b.addRunArtifact(ref_tests);
    const ref_test_step = b.step("test-ref", "Run reference correctness tests");
    ref_test_step.dependOn(&run_ref_tests.step);

    // ==========================================================================
    // Sentence Transformer Benchmark
    // ==========================================================================
    const st_bench_module = b.createModule(.{
        .root_source_file = b.path("tests/benchmark_sentence_transformer.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    st_bench_module.addImport("zblas", zblas_module);

    const st_benchmark = b.addExecutable(.{
        .name = "zblas-benchmark-st",
        .root_module = st_bench_module,
    });

    b.installArtifact(st_benchmark);

    const run_st_benchmark = b.addRunArtifact(st_benchmark);
    const st_bench_step = b.step("bench-st", "Run sentence transformer workload benchmarks");
    st_bench_step.dependOn(&run_st_benchmark.step);

    // ==========================================================================
    // All Benchmarks (convenience target)
    // ==========================================================================
    const bench_all_step = b.step("bench-all", "Run all benchmarks (generic + whisper + cache + sentence-transformer)");
    bench_all_step.dependOn(&run_benchmark.step);
    bench_all_step.dependOn(&run_whisper_benchmark.step);
    bench_all_step.dependOn(&run_cache_benchmark.step);
    bench_all_step.dependOn(&run_st_benchmark.step);
}
