import torch
import IPython

def exists(val) -> bool:
    """
    Checks to see if a value is not `None`
    """
    return val is not None

def default(val, d):
    """
    Returns the input value `val` unless it is `None`, in which case the default `d` is returned if it is a value or
        `d()` is returned if it is a callable.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a: torch.tensor, t: torch.tensor, x_shape: torch.Size) -> torch.tensor:
    """
    Extracts values from `a` using `t` as indices
    :param a: 1D tensor of length L.
    :param t: 1D tensor of length b.
    :param x_shape: Tensor of size (b, c, h, w).
    :return: Tensor of shape (b, 1, 1, 1) that selects elements of a, using t as indices of selection.
    """
    b, *_ = t.shape
    out = a.gather(-1, t) # index into a tensor along a specific dimension
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))