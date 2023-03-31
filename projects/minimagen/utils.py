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


def log(t: torch.tensor, eps: float = 1e-12) -> torch.tensor:
    """
    Calculates the natural logarithm of a torch tensor, clamping values to a minimum of `eps`.
    """
    return torch.log(t.clamp(min=eps))

def prob_mask_like(shape: tuple, prob: float, device: torch.device) -> torch.Tensor:
    """
    For classifier free guidance. Creates a boolean mask for given input shape and probability of `True`.
    :param shape: Shape of mask.
    :param prob: Probability of True. In interval [0., 1.].
    :param device: Device to put the mask on. Should be the same as that of the tensor which it will be used on.
    :return: The mask.
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob