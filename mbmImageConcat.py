# MBM's Music Visualizer: Image Concatenator
# Concatenates ComfyUI image set tensors into a single set tensor.

# Imports
import torch

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
                "set_1": ("IMAGE", ),
                "set_2": ("IMAGE", )
            }
        }

    def process(self, set_1: torch.Tensor, set_2: torch.Tensor) -> torch.Tensor:
        return (torch.cat((set_1, set_2), dim=0), )
