"""tools/ple_rocmfpx_pack.py - a ROCmFPX 3-bit n-gram table as ONE GGUF tensor the engine reads.

    python tools/ple_rocmfpx_pack.py --gguf <model.gguf> --out <ple-rocmfpx.gguf>

A ROCmFPX-quantized Qwen4Exp file keeps the n-gram table as 16 `ple_ngram_embd.<k>.weight` tensors of
[160, vocab_k] Q3_0_ROCMFPX (ggml type 104), 70 bytes a row. The engine's PLE reader wants one
`per_layer_token_embd.weight` of [160, sum(vocab_k)] with contiguous rows, so the 16 payloads are copied in
head order into a one-tensor GGUF. Nothing is decoded or rounded: the bytes are the GGUF's own.
"""
import argparse
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

GGUF_TYPE_STRING = 8
GGML_TYPE_Q3_0_ROCMFPX = 104
PREFIX = "ple_ngram_embd."
ALIGN = 32
CHUNK = 64 << 20


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True, help="the model GGUF holding the ple_ngram_embd.* tensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = pathlib.Path(a.gguf), pathlib.Path(a.out)

    g = GGUFFile(src)
    parts = []
    for t in g.tensors:
        if t.name.startswith(PREFIX) and t.name.endswith(".weight"):
            parts.append((int(t.name[len(PREFIX):].split(".")[0]), t))
    if not parts:
        sys.exit("no %s<N>.weight tensors in %s" % (PREFIX, src))
    parts.sort(key=lambda p: p[0])
    if [p[0] for p in parts] != list(range(len(parts))):
        sys.exit("the tables are not numbered 0..%d without gaps" % (len(parts) - 1))

    dim, rows = None, 0
    for k, t in parts:
        if t.type_id != GGML_TYPE_Q3_0_ROCMFPX:
            sys.exit("%s is type %d, not Q3_0_ROCMFPX (%d)" % (t.name, t.type_id, GGML_TYPE_Q3_0_ROCMFPX))
        if len(t.shape) != 2 or t.shape[0] != 160:
            sys.exit("%s has shape %s, not [160, rows]" % (t.name, t.shape))
        if dim is None:
            dim = t.shape[0]
        elif t.shape[0] != dim:
            sys.exit("%s: inconsistent shape %s" % (t.name, t.shape))
        rows += t.shape[1]
    row_bytes = (dim // 32) * 14                       # Q3_0_ROCMFPX is 14 bytes per 32 values
    print("%d tables, %d rows x %d, %d B/row, %.2f GB"
          % (len(parts), rows, dim, row_bytes, rows * row_bytes / 1e9), flush=True)

    kvs = [kv_string("general.architecture", "strata-ple"),
           kv_string("general.name", "PLE n-gram table, Q3_0_ROCMFPX as quantized"),
           kv_string("strata.ple.source", src.name)]
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
    head += gguf_string("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, rows)
    head += struct.pack("<I", GGML_TYPE_Q3_0_ROCMFPX) + struct.pack("<Q", 0)
    head += b"\0" * ((-len(head)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    written = 0
    with open(src, "rb") as f, open(tmp, "wb") as w:
        w.write(head)
        for k, t in parts:
            n = t.expected_bytes()
            f.seek(g.data_start + t.offset)
            left = n
            while left:
                b = f.read(min(CHUNK, left))
                if not b:
                    sys.exit("short read in %s" % src)
                w.write(b)
                left -= len(b)
            written += n
            if k % 4 == 3 or k == len(parts) - 1:
                print("  %2d/%d tables, %.1f GB" % (k + 1, len(parts), written / 1e9), flush=True)
    if tmp.stat().st_size != len(head) + rows * row_bytes:
        sys.exit("size check failed: %d != %d" % (tmp.stat().st_size, len(head) + rows * row_bytes))
    tmp.replace(out)
    print("wrote %s (%d rows x %d Q3_0_ROCMFPX, header %d B)" % (out, rows, dim, len(head)))


if __name__ == "__main__":
    main()
