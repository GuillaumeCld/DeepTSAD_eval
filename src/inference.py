import numpy as np

# -----------------------------
# Numpy helpers (vectorized)
# -----------------------------

def rowwise_mse(A: np.ndarray,  *args) -> np.ndarray:
    """MSE per-row between A and B: shape (n, m) -> (n,)"""
    return np.mean(A , axis=1)
    diff = A - B
    return np.einsum('ij,ij->i', diff, diff) / diff.shape[1]

def squared_pointwise_error_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Elementwise squared error, same shape as inputs."""
    return (A - B) ** 2



def disjoint_pointwise_profile(pw_error: np.ndarray, n: int, block: int = 100) -> np.ndarray:
    """
    Matches original behavior:
      - take every 100th window's error (flattened by ravel on rows)
      - if tail exists, fill with the last window's last 'remain' values
    """
    out = np.zeros(n, dtype=pw_error.dtype)
    remain = n % block
    sampled = pw_error[::block].ravel()
    if remain == 0:
        out[:] = sampled
    else:
        out[:-remain] = sampled
        out[-remain:] = pw_error[-1, -remain:]
    return out


def combined_pointwise_profile(pw_error: np.ndarray, n: int, window: int) -> np.ndarray:
    """
    Vectorized equivalent of the sliding accumulation using scatter-add with np.add.at.
    This avoids writing to read-only views returned by sliding_window_view.
    """
    acc = np.zeros(n, dtype=pw_error.dtype)
    cnt = np.zeros(n, dtype=np.int32)

    # Build indices for each window position: shape (num_windows, window)
    num_windows = n - window + 1
    idx = np.arange(num_windows)[:, None] + np.arange(window)[None, :]

    # Scatter-add contributions and counts
    np.add.at(acc, idx, pw_error)
    np.add.at(cnt, idx, 1)

    return acc / np.maximum(cnt, 1)

def combined_pointwise_profile_median(pw_error: np.ndarray, n: int, window: int) -> np.ndarray:
    """
    Compute the pointwise median across overlapping windows.

    pw_error: shape (num_windows, window) where num_windows == n - window + 1
    Returns array of length n with the median of all contributions that cover each position.
    """
    pw_error = np.asarray(pw_error)
    if pw_error.ndim != 2:
        raise ValueError("pw_error must be 2D (num_windows, window)")

    num_windows = pw_error.shape[0]
    out = np.empty(n, dtype=pw_error.dtype)

    for i in range(n):
        # windows j that cover position i satisfy: j + k = i  with k in [0, window-1]
        j0 = max(0, i - (window - 1))
        j1 = min(i, num_windows - 1)
        if j1 < j0:
            # no contribution (can happen if window > n); set NaN for float, else 0
            if np.issubdtype(pw_error.dtype, np.floating):
                out[i] = np.nan
            else:
                out[i] = 0
            continue
        j = np.arange(j0, j1 + 1)
        vals = pw_error[j, i - j]
        out[i] = np.median(vals)
    return out

def make_sequences_1d(data: np.ndarray, window: int) -> np.ndarray:
    """
    Build overlapping windows explicitly (shape: (n - w + 1, w)) without using sliding_window_view.
    This returns a brand-new, writeable array.
    """
    if window <= 0:
        raise ValueError("window must be >= 1")
    n = data.shape[0]
    if window == 1:
        return data.reshape(-1, 1).copy()
    num_windows = n - window + 1

    out = np.empty((num_windows, window), dtype=data.dtype)
    # Fill each column j with data[j : j + num_windows]
    for j in range(window):
        out[:, j] = data[j : j + num_windows]
    return out
