import torch


class ForwardProcess():
    def sample(self, image, timestep):
        """Sample from the forward process at a specific timestep.
        
        Args:
            image: The image to noise.
            timestep: The timestep to sample at.
        """
        # Add Gaussian noise to the image
        noise = torch.randn_like(image)
        noised_image = image + noise * timestep # TODO: I don't think timestep can be multipled like that but maybe it can at it's simplest?
        return noised_image
