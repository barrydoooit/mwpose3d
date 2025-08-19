import numpy as np

def make_row_affine(R=None, t=None, dtype=np.float32):
    """
    Build a 4x4 homogeneous matrix M for row vectors so that:
        [x y z 1] @ M  -> applies rotation/translation.
    If R is a 3x3 (column-vector rotation), we store R^T in the block
    to match the row-vector dot(..., R.T) usage in your code.
    """
    M = np.eye(4, dtype=dtype)
    if R is not None:
        M[:3, :3] = R.T  # row-vector convention
    if t is not None:
        M[:3, 3] = np.asarray(t, dtype=dtype)
    return M

def compose_into(input_dict, key, A):
    """Accumulate A onto input_dict[key] with post-multiplication (row-vector)."""
    if key not in input_dict or input_dict[key] is None:
        input_dict[key] = np.eye(4, dtype=np.float32)
    input_dict[key] = input_dict[key] @ A

def invert_row_affine(M):
    """
    Invert a row-vector homogeneous matrix:
       M = [[Q, t],
            [0, 1]]
    Returns:
       M_inv = [[Q^T, -Q^T t],
                [  0,      1]]
    For proper rotations Q should be orthonormal.
    """
    M_inv = np.eye(4, dtype=M.dtype)
    Q = M[:3, :3]
    t = M[:3, 3]
    Q_inv = Q.T
    M_inv[:3, :3] = Q_inv
    M_inv[:3, 3]  = -Q_inv @ t
    return M_inv