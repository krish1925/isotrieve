import numpy as np
import warnings
from aecp.matrix import compute_transfer_matrices

def reproduce():
    np.random.seed(42)
    source = np.random.randn(100, 32)
    target = np.random.randn(100, 64)
    
    print(f"Source dtype: {source.dtype}")
    print(f"Source min: {source.min()}, max: {source.max()}")
    
    print("Testing ridge method...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        W_st, W_ts = compute_transfer_matrices(
            source, target, method="ridge", regularization=0.01
        )
        for warning in w:
            print(f"Caught warning: {warning.message}")
            # print(f"File: {warning.filename}, line: {warning.lineno}")

if __name__ == "__main__":
    reproduce()
