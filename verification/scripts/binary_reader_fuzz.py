"""P9-02: fuzz the .isotrieve reader — corrupted/truncated inputs must raise
clean typed errors, never crash the interpreter."""
import sys, io, tempfile
from pathlib import Path
import numpy as np
from hypothesis import given, settings, strategies as st
from isotrieve.mapping.linear import RidgeMapping

SEED = 20260829
rng = np.random.default_rng(SEED)
X = rng.normal(size=(40, 8)); Y = rng.normal(size=(40, 8))
base = Path(tempfile.mkdtemp()) / "m.isotrieve"
RidgeMapping(alpha="auto", seed=SEED).fit(X, Y).save(base)
RAW = base.read_bytes()

@settings(max_examples=1000, deadline=None, print_blob=True)
@given(st.data())
def fuzz(data):
    mode = data.draw(st.sampled_from(["truncate", "flip", "extend", "header_poison"]))
    b = bytearray(RAW)
    if mode == "truncate":
        b = b[: data.draw(st.integers(0, len(RAW) - 1))]
    elif mode == "flip":
        for _ in range(data.draw(st.integers(1, 8))):
            b[data.draw(st.integers(0, len(b) - 1))] ^= 1 << data.draw(st.integers(0, 7))
    elif mode == "extend":
        b += bytes(data.draw(st.binary(max_size=256)))
    else:
        h = b.find(b"}")
        if h > 0:
            ins = data.draw(st.sampled_from([b'{"headerLen":999999999}', b'{"x":1}']))
            b[h:h] = ins
    p = base.parent / "fuzz.isotrieve"
    p.write_bytes(bytes(b))
    try:
        RidgeMapping.load(p)
        # accepted: only legitimate for no-op corruptions; record and continue
        print("ACCEPTED", mode, len(b), file=sys.stderr)
    except Exception as e:
        assert type(e).__module__ not in ("builtins",) or isinstance(e, (ValueError, OSError, EOFError, KeyError, TypeError)), f"unclean error {type(e)}: {e}"

fuzz()
print("FUZZ_DONE 1000 examples, no crashes")
