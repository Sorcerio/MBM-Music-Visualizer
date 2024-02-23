# MBM's Music Visualizer: Image Concatenator
# Concatenates ComfyUI image set tensors into a single set tensor.

# Imports
import torch
from typing import Optional

# Classes
class ImageConcatenator:
    """
    Concatenates ComfyUI image set tensors into a single set tensor.
    """
    # Class Constants
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGES", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Images"

    # Constructor
    def __init__(self):
        pass

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE", )
            },
            "optional": {
                "image_2": ("IMAGE", )
            }
        }

    def process(self, image_1: torch.Tensor, image_2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if image_2 is None:
            return (image_1, )
        else:
            return (torch.cat((image_1, image_2), dim=0), )
