#pragma once
// include/strata/kernels/rocmfp4.hpp - ROCmFP4 weight format (ggml types 100 Q4_0_ROCMFP4 and
// 101 Q4_0_ROCMFP4_FAST), taken from the ROCmFPX ggml this fork builds against.
//
// Both types are 32 elements per block with the same 4-bit packing as Q4_0: byte j holds element j
// in its low nibble and element j+16 in its high nibble. They differ in the code and the scale:
//   * the 4-bit code indexes a signed codebook {0,1,2,3,4,6,8,10} and its negation, not a nibble
//     offset, so a dot product needs a codebook expansion before DP4A;
//   * the scale is a finite unsigned E4M3 byte whose value is half the IEEE reading, one per
//     16-weight half block (18-byte block) or one for the whole 32 (17-byte block).
// Source of truth: ROCmFPX ggml/rocmfp4/rocmfp4.h and rocmfp4.c.

#include <cstdint>

namespace strata::kernels {

struct RocmFp4Block {
    uint8_t qs[16];
    uint8_t e[2];
};
struct RocmFp4FastBlock {
    uint8_t qs[16];
    uint8_t e;
};
static_assert(sizeof(RocmFp4Block) == 18, "block_rocmfp4 is 18 bytes");
static_assert(sizeof(RocmFp4FastBlock) == 17, "block_rocmfp4_fast is 17 bytes");

#if defined(__CUDACC__) || defined(__HIPCC__)

// Finite unsigned E4M3 byte to its half-scale value: (8 + m) * 2^(ex - 11), ex = 0 subnormal.
// Built from exponent and mantissa bits so fast math cannot round it.
__device__ __forceinline__ float rocmfp4_scale(uint8_t e) {
    if (e > 0x7e) return 0.0f;
    const int ex = (e >> 3) & 0x0f;
    const int m = e & 0x07;
    if (ex == 0) return float(m) * 0x1p-10f;
    return __uint_as_float(((uint32_t) (127 + ex - 8) << 23) | ((uint32_t) m << 20));
}

__device__ __forceinline__ int rocmfp4_code(int q) {
    const int mag3 = q & 0x07;
    const int mag = mag3 <= 4 ? mag3 : 2 * mag3 - 4;
    return (q & 0x08) ? -mag : mag;
}

// Four packed nibbles to four signed int8 codebook values, one per byte. A byte at a time, so the
// 17-byte block's odd strides need no alignment.
__device__ __forceinline__ int rocmfp4_codebook_pack4(int x) {
    int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r |= (rocmfp4_code((x >> (8 * i)) & 0x0f) & 0xff) << (8 * i);
    }
    return r;
}

// A 4-byte code word of a block, assembled byte by byte (the fast block's stride is odd).
__device__ __forceinline__ int rocmfp4_load_i32(const void* qs, int i) {
    const uint8_t* p = (const uint8_t*) qs + 4 * i;
    return (int) p[0] | ((int) p[1] << 8) | ((int) p[2] << 16) | ((int) p[3] << 24);
}

// ---- The rest of the ROCmFPX family (ggml types 102-111) -----------------------------------------
//
// Every type is 32 values per block and shares the scale decode above: the two 16-element halves of a
// block each carry one scale byte (e[0], e[1]), or one byte (e) covers the whole block. Only the code
// packing and the code meaning differ, and there are two packings:
//   * "split halves" (Q4_0-style): code[i] for i < 16 is the low nibble of qs[i], for i >= 16 the high
//     nibble of qs[i-16]. Used by Q4_0_ROCMFP4/_FAST (codebook10) and Q4_0_ROCMI4 (signed nibble).
//   * "bit stream" (LSB first): code[i] is the `bits`-wide field starting at bit i*bits of qs. Used by
//     Q2_0_ROCMFPX (2), Q3_0_ROCMFPX (3), Q5_0_ROCMFPX (5), Q6_0_ROCMFPX (6), Q7_0_ROCMFPX (7),
//     Q8_0_ROCMFPX (8).
// Source of truth: ROCmFPX ggml/rocmfpx/rocmfpx.c and rocmfpx.h.

// Block geometry of the family: (qs bytes, scale bytes, bits per code, packing).
// packing 0 = bit stream, 1 = split halves (nibble), 2 = split halves (signed nibble).
struct RocmFpxFormat {
    int qs_bytes;
    int scales;
    int bits;
    int packing;
};

// false when the type is not one of the family this header decodes.
__host__ __device__ inline bool rocmfpx_format(int type, RocmFpxFormat& f) {
    switch (type) {
    case 100: f = {16, 2, 4, 1}; return true;   // Q4_0_ROCMFP4
    case 101: f = {16, 1, 4, 1}; return true;   // Q4_0_ROCMFP4_FAST
    case 102: f = {24, 2, 6, 0}; return true;   // Q6_0_ROCMFPX
    case 103: f = {32, 1, 8, 0}; return true;   // Q8_0_ROCMFPX
    case 104: f = {12, 2, 3, 0}; return true;   // Q3_0_ROCMFPX
    case 107: f = {8, 2, 2, 0}; return true;    // Q2_0_ROCMFPX_LEGACY_AMBIGUOUS
    case 108: f = {16, 1, 4, 2}; return true;   // Q4_0_ROCMI4
    case 109: f = {20, 2, 5, 0}; return true;   // Q5_0_ROCMFPX
    case 110: f = {28, 2, 7, 0}; return true;   // Q7_0_ROCMFPX
    case 111: f = {8, 2, 2, 0}; return true;    // Q2_0_ROCMFPX
    default: return false;
    }
}

__device__ __forceinline__ int rocmfpx_twos_complement_nibble(uint8_t v) {
    v &= 0x0f;
    return (v & 0x08) ? (int) v - 16 : (int) v;
}

// One code -> its integer value. `rocmfp4_code` is the codebook10 used by types 100/101/108's sibling.
__device__ __forceinline__ int rocmfpx_code_value(int type, uint8_t code) {
    switch (type) {
    case 100:
    case 101: {
        const int mag3 = code & 0x07;
        const int mag = mag3 <= 4 ? mag3 : 2 * mag3 - 4;
        return (code & 0x08) ? -mag : mag;
    }
    case 102: {                                    // fp6: mag in 0..31, negative zero reads as 32
        const int mag = code & 31;
        return (code & 32) ? -(mag == 0 ? 32 : mag) : mag;
    }
    case 103:
        return (int) (int8_t) code;                // fp8: a plain signed byte
    case 104: {                                    // fp3: mag in {0,1,2,4}
        const int mag = (int) (code & 3u);
        const int v = mag == 0 ? 0 : mag == 1 ? 1 : mag == 2 ? 2 : 4;
        return (code & 4u) ? -v : v;
    }
    case 107:
    case 111: {                                    // fp2: the frozen MORD order
        static const int values[4] = {-4, -1, 1, 4};
        return values[code & 3u];
    }
    case 108:
        return rocmfpx_twos_complement_nibble(code);
    case 109:
    case 110: {                                    // linear sign-magnitude
        const int bits = type == 109 ? 5 : 7;
        const int sign = 1 << (bits - 1);
        const int mag = code & (sign - 1);
        return (code & sign) ? -(mag == 0 ? sign : mag) : mag;
    }
    default:
        return 0;
    }
}

// The i-th code of a block (i in 0..31).
__device__ __forceinline__ uint8_t rocmfpx_block_code(const uint8_t* qs, const RocmFpxFormat& f, int i) {
    if (f.packing == 0) {                          // LSB-first bit stream
        const int bit = i * f.bits, byte = bit >> 3, shift = bit & 7;
        uint16_t packed = qs[byte];
        if (shift + f.bits > 8) packed |= (uint16_t) qs[byte + 1] << 8;
        return (uint8_t) ((packed >> shift) & ((1u << f.bits) - 1u));
    }
    return (i < 16) ? (uint8_t) (qs[i] & 0x0f) : (uint8_t) (qs[i - 16] >> 4);
}

// The scale of element i (0..31): the block's own two halves, or one byte for all 32.
__device__ __forceinline__ float rocmfpx_block_scale(const uint8_t* blk, const RocmFpxFormat& f, int i) {
    return f.scales == 1 ? rocmfp4_scale(blk[f.qs_bytes])
                         : rocmfp4_scale(blk[f.qs_bytes + (i < 16 ? 0 : 1)]);
}

// A Q8_1 block's leading fp16 scale, decoded without pulling in cuda_fp16 (this header is included
// before it in some translation units).
__device__ __forceinline__ float rocmfpx_q8_scale(const uint8_t* qb) {
    const uint32_t h = (uint32_t) qb[0] | ((uint32_t) qb[1] << 8);
    const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    if (exp == 31) return 0.0f;                      // inf/nan: never a real scale
    const float mag = exp == 0 ? (float) man * 0x1p-24f
                               : __uint_as_float(((exp - 15 + 127) << 23) | (man << 13));
    return sign ? -mag : mag;
}

// The dot product of one family block with one Q8_1 block (fp16 scale followed by 32 int8 values, the
// layout both Q81Block and ggml's block_q8_1 have). Correct for every type in the family; the
// per-format kernels in native_mmvq.cu / iq_kernels.cu are the faster specializations of it.
__device__ __forceinline__ float rocmfpx_block_dot(int type, const uint8_t* blk, const void* q8blk) {
    RocmFpxFormat f{};
    if (!rocmfpx_format(type, f)) return 0.0f;
    const uint8_t* qb = (const uint8_t*) q8blk;
    const int8_t* q8 = (const int8_t*) (qb + 4);
    const float d8 = rocmfpx_q8_scale(qb);
    if (f.scales == 1) {
        int s = 0;
        for (int i = 0; i < 32; ++i) s += rocmfpx_code_value(type, rocmfpx_block_code(blk, f, i)) * (int) q8[i];
        return d8 * rocmfp4_scale(blk[f.qs_bytes]) * (float) s;
    }
    int s0 = 0, s1 = 0;
    for (int i = 0; i < 16; ++i) s0 += rocmfpx_code_value(type, rocmfpx_block_code(blk, f, i)) * (int) q8[i];
    for (int i = 16; i < 32; ++i) s1 += rocmfpx_code_value(type, rocmfpx_block_code(blk, f, i)) * (int) q8[i];
    return d8 * (rocmfp4_scale(blk[f.qs_bytes]) * (float) s0 +
                 rocmfp4_scale(blk[f.qs_bytes + 1]) * (float) s1);
}

#endif

}  // namespace strata::kernels
