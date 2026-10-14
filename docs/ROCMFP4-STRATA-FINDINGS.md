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
| 4K | **905 tok/s** | 808.5 | **112%** | **71.91 tok/s** | ~50 | **144%** | 2.24x |
| 8K | **982 tok/s** | 1159 | 85% | **71.19 tok/s** | 53.14 | **134%** | 2.51x |
| 32K | **1021 tok/s** | 1290 | 79% | **67.72 tok/s** | 50.91 | **133%** | 3.13x |

Initial decode was **5.80 tok/s**; it is 67.7-71.9 now, about 12x.

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
# gfx1151's default table now also carries STRATA_PF_HCDOWN (see 3)        -16%
```

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

## 5. What was measured and rejected (do not re-try these)

- **Switches, all neutral or worse:** `GDN_CHUNKED`, `GDN_CP_ASYNC`, `GDN_PIPELINE`, `GDN_REC_HEADS`,
  `GDN_CONV_SERIAL`, `PF_OCC` 1/2/4, `PF_BK64=0`, `PF_FUSED_TILE` 64/128, `PF_FUSED_NATIVE`, `PF_STREAM_MIN`
  128/1024, `GR_UNFUSED`, `EMB_REUSE_ACCOUNT`, `STREAM_AHEAD`, `STAGER_SLEEP`, `FUSED_RING`, `W_LB` 0/128/168/192,
  `NAT_WMMA`, `FUSED_W11` 0/1, `AUX_STAGER`, `GROUP_COPY`, `BF16_TC`, `INDEXER_PER_TOKEN`, `ATOMIC_WAIT`,
  `KV_PREFETCH`, `PROMPT_ATTN_OLD`, `SELECT_OLD`, `SELECT_SIMT`, `TOPK_OLD`, `TOPK_STREAM`, `STATE_HASH_GDN`,
  `DENSE_MMQ`, `ORCA_Q4KS_MMQ` (compile-time, Q5_0 only), `--no-prefill-borrow`, `--prefill` 4096/8192/16384/32768,
  the hipBLASLt tuning table, `--kv k8v4` / `q4_0`, `--mtp-window` 2048/8192/16384, `--mtp-hnorm` pooled,
  spec 3/5/6/8, `--spec-min-p` 0.2-0.6, `--ple-io direct`.
- **Two "speedups" that are wrong and must not be enabled:**
  - `STRATA_HIP_PROMPT_F16=1` is 7% faster on the GPU timeline and **degrades the output** to
    `220 735 220 735 ...`.
  - `STRATA_WMMA_GEMM=1` shows -6.6% on the GPU timeline but a worse and far more variable wall clock
    (800-1042 against 1002-1015). Per this project's own rule, wall clock decides: rejected.
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
