# ROCmFP4 on Strata — what was found, what was changed, what was measured

Machine: gfx1151 (Strix Halo, Ryzen AI Max+ 395), 96 GiB VRAM carve-out + 30 GB system, ROCm 10.0.
Model: `Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf` (87.05 GiB, `qwen4exp`, Q4_0_ROCMFP4_FAST down experts,
Q4_K gate/up, Q6_K dense, 16 `ple_ngram_embd.N` tensors).
Reference: the same file under ROCmFPX (commit `26ca4e2cd`, Vulkan backend; its HIP backend cannot load it).
All numbers below are wall clock on this machine unless a GPU timeline is named, three runs where it matters.

## 1. Where it started and where it is

Decode is quoted from a **256-token** generation; a 32-token one reads 6-10% lower because of warm-up, which is
easy to mistake for a configuration effect.

| depth | prefill | official "fast" | vs official | decode | official | vs official | vs ROCmFPX |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | **959.78 tok/s** | 808.5 | **119%** | **66.21 tok/s** | ~50 | **132%** | 2.37x |
| 8K | **1038.78 tok/s** | 1159 | **90%** | **65.45 tok/s** | 53.14 | **123%** | 2.66x |
| 32K | **1066.27 tok/s** | 1290 | **83%** | **64.65 tok/s** | 50.91 | **127%** | 3.27x |

Three runs each. Decode is quoted from a **128-token** generation; a 256-token one reads a few percent higher
and a 32-token one 6-10% lower, because the warm-up is a fixed cost - enough to turn a 6% configuration effect
into a wrong answer, which is how `STRATA_WMMA_GEMM` came to be rejected once (section 5).

Initial decode was **5.80 tok/s**.

The initial 5.8 tok/s was three stacked configuration errors, not ROCmFP4:

1. `--expert-cache 4096` left 83% of the 24576 experts outside VRAM, so most routing went to the CPU pool
   (224 ms of a 250 ms round). `--expert-cache auto` gives 31288 slots and the pool drops to 0.000 ms/round.
2. The MTP drafter was never loaded, so `--spec 4` verified 6 tokens per round and emitted 1 (acceptance 0 of 93).
3. The fused prompt-expert path and `STRATA_PF_GEMM` were off.

## 2. The configuration that matters

```bash
export STRATA_PF_FUSED=1        # the fused prompt experts themselves
export STRATA_PF_FUSED_KQ=1     # Q4_K gate/up coverage (HIP only)
export STRATA_PF_FUSED_FPX=1    # ROCmFPX down coverage (new code, see 3)
export STRATA_PF_GEMM=1         # WMMA dense GEMM instead of hipBLASLt      -45% of the chunk
export STRATA_HIP_WMMA=1        # prompt attention on matrix cores          -6.3%
export STRATA_HC_UPMIX=1        # hyper-connection upmix                    -6.5%
# gfx1151's default table now also carries STRATA_PF_HCDOWN (see 3b)       -16%
# and #313's STRATA_WMMA_GEMM (see 3d)                          +5.6..+9.1%
```

`STRATA_PF_GEMM` and `STRATA_WMMA_GEMM` are **both** needed: the S23 fused path runs first and returns, and
the WMMA one covers the shapes that path declines, falling through to hipBLASLt when it declines them too.

Also required, and outside the switches: `--ple-io mmap` (page cache; `direct` costs 23% of decode),
`--expert-cache auto`, `--mtp <rt dir>`, `--mtp-draft-vocab data/draft_vocab.bin` (+6.7% decode),
`--spec 4 --spec-min-p 0.35`, `--kv int8`, `--mtp-q4 all`, `--lookup-chain 3`.

**Do not set `STRATA_PF_SWITCH_MIN_T`.** It gates the rounding-level prompt switches below a token count;
set to 4096 it silently reverts every one of them for a 4095-token prompt, which cost 2x before it was spotted.

`--kv int8` is the right KV type for prefill too, not only decode: at 32K, `k8v4` is 22% slower and `q4_0` 20%.

## 3. The three code changes

### 3a. `src/prefill/moe_fused_iq.cu` - ROCmFP4 (types 100/101) in the fused prompt-expert path (+11%)

The fused prompt experts covered gate/up IQ2_*/IQ3_*/IQ4_XS, and down Q2_0 / IQ4_NL only. A pack whose down is
Q4_0_ROCMFP4_FAST therefore could not use the path at all. Added: the type constants, `block_bytes` (18 / 17),
`raw_words`, a `d_pair` group for the down addressing, the `load_unit` case, the `convert<>` branch (split-halves
nibble through `rocmfp4_codebook_pack4`, UE4M3 scale through `rocmfp4_scale`), `d_covered`, and the dispatch.

**Two things cost four rounds here, both worth knowing:**

- **There are two launch paths, and they return before each other.** `experts_native` takes an early-return path
  through `native_w11_kernel` (the `STRATA_NW_GU` / `STRATA_NW_D` chains) on HIP, and only reaches the
  `launch<>` switch on CUDA. Adding the type to the `launch<>` switch alone changes nothing on this machine -
  the model decoded to **all zeros**, because the `STRATA_NW_D` chain had no ROCmFP4 case and fell through to
  `IQ4_NL`, reading a 17-byte block as an 18-byte one. The proof that the fix had landed was a *compile error*
  (`reference to __host__ function 'getenv' in __device__ function`) from a debug statement that had been
  sitting in the dead branch: only then was the device code actually instantiated.
- **`ld16(p)` is `*(const uint16_t*) p` - a bare 16-bit dereference, so it assumes 2-byte alignment.** Every
  other format's block is an even number of bytes (18/24/34/66/98/110/136/144/176), so the assumption had never
  been broken. `Q4_0_ROCMFP4_FAST`'s **17-byte** block puts `(2s + uj) * BS` on odd addresses. The ROCmFP4
  `load_unit` uses a byte-wise load for that reason.

Correctness evidence: the decoded tokens are **byte-identical** with and without the path, over 8+ runs at 8K
and 32K, and `native_expert_parity` (0 failures) and `hip_prefill_mmq_parity` (screen passed, rel_l2 6.6e-4)
both pass with it enabled.

### 3b. `src/core/arch_defaults.cpp` - `STRATA_PF_HCDOWN=1` in the gfx1151 defaults (+16%)

The prompt phase timer (`STRATA_PREFILL_TIMING=1`) showed `hc read` at **26.5%** of the chunk. The table already
carried `STRATA_HCD_EXACT`, but that switch is **inert on this machine** ("no hipBLASLt tuning table is loaded,
so the exact kernel has nothing to match: the hyper-connection down projection stays on hipBLAS" - and loading
`tools/hip/gfx1151-hipblaslt-100401.txt` does not change `hc read`, because `PF_HCDOWN` replaces that path
anyway). `STRATA_PF_HCDOWN` fuses the down and inject projections into one GEMM:

```
hc read        2600 -> 610 ms
GPU timeline   9543 -> 8034 ms   (-15.8%, all three runs)
```

The output stays byte-identical, so it belongs in the defaults; reverse control with
`STRATA_GFX1151_DEFAULTS=0`: 10654-10682 ms against 8156-8476 ms.

### 3c. `src/prefill/prefill.cpp` - `STRATA_PF_STEP_MIN_MS` (diagnostics)

`STRATA_PF_STEP_SYNC`'s report threshold was a hard 250 ms. At 8192 tokens a gfx1151 layer takes ~229 ms, so
**every step was filtered and the diagnosis printed nothing at all**. The threshold is now a knob; the phase
timer (`STRATA_PREFILL_TIMING=1`) is the better tool and already existed.

### 3d. `src/core/arch_defaults.cpp` - `STRATA_WMMA_GEMM=1` in the gfx1151 defaults (+5.6..+9.1%)

`Gemm::native`'s HIP path tries, in order: the S23 fused prompt GEMM (`STRATA_PF_GEMM`), then #313's WMMA GEMM
(`STRATA_WMMA_GEMM`), then `f16_inplace`, then hipBLASLt. The first one returns when it handles the shape, so
the second covers what it declines - which is why both are needed and why turning the second on is worth
another 5.6-9.1% *on top of* `PF_GEMM`. It is try-then-fall-through, so it cannot make an unsupported shape
fail.

    prompt @4096   910.21 -> 979.96 tok/s   (+7.7%)
    prompt @8192   943.61 -> 1029.21 tok/s  (+9.1%)
    prompt @32768 1007.51 -> 1064.14 tok/s  (+5.6%)
    decode         unchanged            decoded tokens identical over 128 steps

Reverse control with `STRATA_GFX1151_DEFAULTS=0`: 732.40 / 744.97 tok/s at 8192 against 1026.84 / 1033.74 /
1029.65 with the table.

## 4. The phase breakdown at 8192 tokens (GPU timeline 7792 ms)

| phase | ms | % |
| --- | --- | --- |
| gemm gate/up (the whole fused expert call) | 2481 | 31.8% |
| gdn (all four parts) | 2073 | 26.6% |
| qsa (proj + attn + select) | 1153 | 14.8% |
| combine | 731 | 9.4% |
| hc read | 610 | 7.8% |
| router + shared | 502 | 6.4% |
| ple | 153 | 2.0% |

`combine` is ~2.5 GB of traffic per layer (read `Dm` + read `shared` + write `bo`), i.e. ~560 ms per chunk at
215 GB/s - it is already near the bandwidth wall, not 4x off it. `gemm gate/up` and `gdn` are the two large
phases left and neither has a runtime knob; getting at them means kernel work.

### 4a. Where the fused expert call goes, and why gate/up cannot be tuned

`STRATA_PF_SPLIT_TIMING=1` splits the one `experts_native()` call into its two kernels:

    gate/up  1683.8 ms      down  818.8 ms      (8192 tokens, 48 layers, cumulative)

A 2:1 split, which is the FLOP ratio. gate/up is the single largest item of the whole prompt chunk (21.6% of
its 7792 ms timeline). `STRATA_PF_KERNEL_INFO=1` then gives the resources:

    Q4_K gate/up        256 VGPRs (the RDNA3 maximum), LDS 41472 B -> 1 block per WGP
    Q4_K gate/up ALIAS  256 VGPRs,                     LDS 24832 B -> 2 blocks per WGP
    ROCmFP4 down        240 VGPRs,                     LDS 24832 B -> 2 blocks per WGP

gate/up is pinned to one resident block because 41472 B leaves no room for a second in the WGP's 64 KB of
LDS. **But doubling the resident blocks does not help**: the ALIAS variant really does reach two, and gate/up
was still 3.7% SLOWER with it (2572 against 2481 ms) - the extra barrier costs more than the occupancy returns.
This kernel is therefore **not occupancy-bound**, and neither the LDS budget nor the VGPR cap is the lever.

**Do not price this kernel against a DP4A peak.** An earlier pass called it "15% of the int8 rate, 6.7x off",
using the ~98 TOPS that `v_dot4` implies. This kernel does not execute `v_dot4`: it is `nw_wmma` + `ldmatrix`
(16x16x16 tiles), whose int8 rate on RDNA3 is a different and much lower number. The measured figure is
**1.29e13 MACs per chunk in 1684 ms = 7.7 TMAC/s = 15.4 TOP/s**, and how close to the WMMA issue limit that is
has not been established here. Treat the "6.7x of peak" claim as withdrawn.

The inner loop explains the occupancy results: for the types that are not `per16` - which includes Q4_K - the
two WMMA steps are **chained**, `nw_wmma(A1, B1, nw_wmma(A0, B0, m))`, deliberately, so that only one `nw_i8`
accumulator is live. The `per16` types (IQ2_XS / IQ2_S) use two independent WMMAs and a second accumulator. So
trading registers for instruction-level parallelism is available, but the register file is already full (256
VGPRs, 32 for the accumulators, 64 for the two prefetch buffers, 16 for the scales), which is why every cap in
the 2..8 range measured worse: the compiler does not give back VGPRs, it spills. What is left is the per-stage
`__syncthreads()` (gate/up runs **40** 64-value stages) and the dependency chain - both need the pipeline
rearranged, not a knob.

**The barrier is not it, though.** The two halves of the same call have very different stage counts and the
same rate:

| half | ms | MACs | rate |
| --- | --- | --- | --- |
| gate/up (40 stages) | 1683.8 | 1.288e13 | **7.65 TMAC/s** |
| down (10 stages) | 818.8 | 6.44e12 | **7.87 TMAC/s** |

Four times the barriers, identical throughput. Both use the same `nw_wmma` int8 form, and both run at about
**half** the rate the S23 dense GEMM's own comment quotes for gfx1151 (30-33 TFLOPS on the 16K-row shapes =
15-16.5 TMAC/s). What the two expert halves share and the dense GEMM does not is the epilogue: per WMMA,
`dotf()` (an int32-to-float convert of the accumulator), a `w0[i] * dx` multiply and an FMA, doubled again for
the `has_min` types - roughly 12 to 16 ALU operations per WMMA. That ratio, not occupancy and not the barrier,
is where the remaining ~2x would come from, and it is an epilogue rewrite rather than a tile change.

Paths already measured for gate/up, so nobody repeats them:

| path | result |
| --- | --- |
| occupancy variant + ALIAS with `-DSTRATA_W_LB=8` | **+68%** (2749-2802 ms) - the VGPR cap spills, not ALIAS |
| ALIAS alone (`NW_LB=1`) | +3.7% (2572 against 2481), despite occupancy 1 -> 2 |
| grid factor `STRATA_PF_OCC` 2 against 4 | no difference on the whole chunk (7976 against 7987 ms) |
| `NW_ROWS` 128 -> 64 (halves the weight-buffer LDS) | does not compile: the ALIAS static_assert wants `kTileRows * 65 * 4 <= 2 * NW_ROWS * WLD`, 64 * 260 > 10240 |
| `STRATA_PF_OCC_W` 1/2/3/6 | 10140 / 7957 / 8091 / 8005 ms against 8051 - neutral or worse |
| `STRATA_W_LB` | **a compile-time define, not an environment variable** - passing it as one is a no-op (measured wrong once, corrected) |

`gdn`'s 26.6% is mostly dense GEMM too, not the recurrence: of its 2073 ms the in-projection (`wqkv`) is
1119 ms and the out-projection (`wo`) 432, with conv+gates 124 and the recurrence 398. Those go through the
WMMA dense path `STRATA_PF_GEMM` already turns on.

## 5. What was measured and rejected (do not re-try these)

- **Switches, all neutral or worse:** `GDN_CHUNKED`, `GDN_CP_ASYNC`, `GDN_PIPELINE`, `GDN_REC_HEADS`,
  `GDN_CONV_SERIAL`, `PF_OCC` 1/2/4, `PF_BK64=0`, `PF_FUSED_TILE` 64/128, `PF_FUSED_NATIVE`, `PF_STREAM_MIN`
  128/1024, `GR_UNFUSED`, `EMB_REUSE_ACCOUNT`, `STREAM_AHEAD`, `STAGER_SLEEP`, `FUSED_RING`, `W_LB` 0/128/168/192,
  `NAT_WMMA`, `FUSED_W11` 0/1, `AUX_STAGER`, `GROUP_COPY`, `BF16_TC`, `INDEXER_PER_TOKEN`, `ATOMIC_WAIT`,
  `KV_PREFETCH`, `PROMPT_ATTN_OLD`, `SELECT_OLD`, `SELECT_SIMT`, `TOPK_OLD`, `TOPK_STREAM`, `STATE_HASH_GDN`,
  `DENSE_MMQ`, `ORCA_Q4KS_MMQ` (compile-time, Q5_0 only), `--no-prefill-borrow`, `--prefill` 4096/8192/16384/32768,
  the hipBLASLt tuning table, `--kv k8v4` / `q4_0`, `--mtp-window` 2048/8192/16384, `--mtp-hnorm` pooled,
  spec 3/5/6/8, `--spec-min-p` 0.2-0.6, `--ple-io direct`.
- **One "speedup" that is wrong and must not be enabled:** `STRATA_HIP_PROMPT_F16=1` is 7% faster on the GPU
  timeline and **degrades the output** to `220 735 220 735 ...`.
- **`STRATA_WMMA_GEMM` was rejected once and should not have been.** The disqualifying measurement ran with
  `--max-new 16`, and at that length the difference read as -6.6% on the GPU timeline but a *worse and far more
  variable* wall clock (800-1042 against 1002-1015). Re-measured at `--max-new 64` with five runs a side, the
  GPU timelines separate cleanly (7955..8039 against 7482..7708 ms) and the wall clock agrees at 4096, 8192 and
  32768 tokens (+7.7%, +9.1%, +5.6%) with identical 128-token output. It is now in the gfx1151 defaults (3d).
  **The lesson is the one already in section 1: quote decode from a 128-token generation.**
- **A second sweep, of thirty-two variables neither of the earlier sweeps had touched**, found nothing either.
  Every one was set to 1 against a 7527 ms baseline, GPU timeline, 8192 tokens:

  | variable | ms | | variable | ms |
  | --- | --- | --- | --- | --- |
  | `GR_FAST` | 7492 | | `QSA_EARLY_EXIT` | 7517 |
  | `GR_NORM_FAST` | 7502 | | `QSA_CLUSTER` | 7574 |
  | `GR_SPLIT` | 7549 | | `QSA_WARP` | 7521 |
  | `GR_DOWN_MAX4` | 7622 | | `PROMPT_ATTN_Q4` | 7552 |
  | `GR_V3` | 7743 | | `PROMPT_ATTN_V1` | 7527 |
  | `GR_UNFUSED` | **9495** | | `SELECT_WMMA` | 7606 |
  | `HC_Q8` | 7552 | | `GDN_NOY` | 7531 |
  | `HC_Q8_INJECT` | 7472 | | `GDN_CONVL2` | 7695 |
  | `HC_SPLIT` | 7613 | | `GDN_SPLIT` | 7587 |
  | `QFUSE` | 7740 | | `GDN_HEAD` | 7639 |
  | `QFUSE_HC` | 7638 | | `GDN_KEYHEAD` | 7546 |
  | `QFUSE_HC_LOCAL` | 7599 | | `GDN_COMMIT_SPLIT` | 7552 |
  | `FUSE_CONVAB` | 7504 | | `KQ256` | 7551 |
  | `FUSE_HEAD_GR` | 7495 | | `ROUTER_BLOCK` | 7524 |
  | `FUSE_HITS` | 7575 | | `HIP_ROUTER_FAST` | 7804 |
  | `FUSE_WAIT` | 7562 | | `GR_UNFUSED` | (see above) |

  The apparent leader, `HC_Q8_INJECT` at -0.7%, does not survive three runs a side: 7488/7516/7581 without it
  against 7487/7676/7683 with it. Nothing else is outside a +-50 ms noise band either. **Treat the
  configuration line as closed** - both the variables that are reachable and the ones that are not.
- **`STRATA_PF_FUSED_KQ` / `STRATA_PF_FUSED` alone look useless if the expert path is not already on.**
  Measuring them with `PF_FUSED` unset put the experts on MMQ, which dominated the chunk and hid every other
  switch's effect (21397 against 22372 ms, apparently "all useless"). The 6 worthwhile switches were all found
  only after the ordering was fixed. **Turn the expert path on first, then sweep.**
- **The official numbers are not reproducible on this machine by the reference implementation.** ROCmFPX, on
  the official `UD-IQ4_XS` file itself, with `GGML_VK_ROCMFP4_COOPMAT=1` and `-ctk/-ctv q8_0`: pp4096 341.10,
  pp8192 340.63, pp32768 298.15, tg128 24.08 - against the documented 808.5 / 1159 / 1290 / 53.14. The BIOS
  split matches the documentation (kernel `mem_info_vram_total` = 96 GiB, system 30 GB), so the difference is
  not a memory configuration.
- **The residual 12-15% of prefill is engine-wide, not quantisation.** With the same configuration, Strata on
  `UD-IQ4_XS` and on this ROCmFP4 file reach the **same 88%** of the official 8K number (1022 and 1014 tok/s).
  A gate/up quantisation difference would have separated them; it does not.

## 6. One caveat on the recommended configuration

`STRATA_PF_GEMM`, `STRATA_HIP_WMMA` and `STRATA_HC_UPMIX` are the S23 **rounding-level prompt switches** that
`STRATA_PF_SWITCH_MIN_T` exists to gate. They are worth 2.25x once the fused expert path is on (399.68 against
898.42 tok/s at 4K in the same run), and the decoded tokens were byte-identical in every test done here - but
they are opt-ins for a reason, and enabling them for very short prompts changes numerics the maintainers chose
to keep. `STRATA_PF_HCDOWN` was promoted to a default instead because it is not one of those switches and its
output is byte-identical.

## 7. Landing this branch

The work is on `fork/rocmfpx-family` (seven commits on top of `a0116b27`). Merging it into the fork's current
`main`, which is 282 upstream commits ahead (0.1.44.1), conflicts in **two files, four blocks**:

| file | what collides |
| --- | --- |
| `src/kernels/cuda/iq_kernels.cu` | upstream added NVFP4 (type 40) to `STRATA_GU_FMTS` / `STRATA_D_FMTS` / `STRATA_MMVQ_FMTS`; this branch added 100/101 and the family. Keep both. |
| `src/prefill/moe_mmq.cu` | upstream added Q6_K / the dense paths to the MMQ support list; this branch added the ROCmFPX family. Keep both. |

The other 15 files merge cleanly. The resolution is mechanical, **but it must be followed by a rebuild and a
re-verification**: both sides are compile-time macro lists, so a dropped type does not fail the build - it takes
a silent wrong branch at run time, which is exactly how a missing `STRATA_NW_D` case decoded this model to
zeros. Everything in this document was measured on `a0116b27`, not on those 282 commits.

Suggested order: merge, resolve the four blocks keeping both sides, build, run `native_expert_parity` and
`hip_prefill_mmq_parity`, then a 64-token `--tokens-file` run compared against the token ids in section 3a.

The two `docs/` files added by this branch are this document and nothing else; the code changes are the three
in section 3 plus the four diagnostics named in 3c and 4a.
