import torchvision
import torch

def save_image(tensor: torch.Tensor, path: str) -> None:
    """Saves an image tensor to a specified path.

    Args:
        tensor (torch.Tensor): The image tensor to save. Expected to be in (C, H, W) format.
        path (str): The path where the image will be saved.
    """
    torchvision.utils.save_image(tensor, path)

