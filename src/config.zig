// zblas/src/config.zig
// Compile-time configuration and tuning parameters for zblas
//
// These values control cache blocking and micro-kernel dimensions.
// Optimal values depend on CPU cache sizes and SIMD width.

const std = @import("std");
const builtin = @import("builtin");

/// Micro-kernel dimensions (MR x NR block computed per kernel call)
/// These must match the actual kernel implementations
pub const MR = switch (builtin.cpu.arch) {
    .x86_64 => 8, // AVX/AVX2: 8 floats per YMM register
    .aarch64 => 8, // NEON: 2x4 floats
    else => 4, // Generic: 4 floats per vector
};

pub const NR = switch (builtin.cpu.arch) {
    .x86_64 => 8,
    .aarch64 => 8,
    else => 4,
};

/// Cache blocking parameters (tuned for typical L2/L3 sizes)
/// MC: Rows of A block (fits in L2 with KC columns)
/// KC: K dimension block (shared dimension)
/// NC: Columns of B panel (fits in L3 with KC rows)
pub const MC = switch (builtin.cpu.arch) {
    .x86_64 => 256,
    .aarch64 => 256,
    else => 128,
};

pub const KC = switch (builtin.cpu.arch) {
    .x86_64 => 512,
    .aarch64 => 512,
    else => 256,
};

pub const NC = switch (builtin.cpu.arch) {
    .x86_64 => 4096,
    .aarch64 => 4096,
    else => 1024,
};

/// Memory requirements for packing buffers
/// A buffer: MC * KC * sizeof(f32)
/// B buffer: KC * NC * sizeof(f32)
pub fn getPackedASize() usize {
    return MC * KC;
}

pub fn getPackedBSize() usize {
    return KC * NC;
}

/// Minimum matrix size to use optimized path (smaller uses reference)
pub const MIN_OPTIMIZED_SIZE = 32;

/// Alignment for packed buffers (cache line size)
pub const CACHE_LINE_SIZE = 64;

/// Check if architecture supports SIMD
pub fn hasSimd() bool {
    return switch (builtin.cpu.arch) {
        .x86_64 => true,
        .aarch64 => true,
        .wasm32 => true, // WASM SIMD
        else => false,
    };
}

/// Get the vector width in floats for current architecture
pub fn getVectorWidth() usize {
    return switch (builtin.cpu.arch) {
        .x86_64 => 8, // AVX: 256-bit = 8 floats
        .aarch64 => 4, // NEON: 128-bit = 4 floats
        .wasm32 => 4, // WASM SIMD: 128-bit = 4 floats
        else => 1, // Scalar fallback
    };
}
