# MBM's Music Visualizer: Shared
# Shared functionality for the Music Visualizer.

# Imports
import os
import torch
import numpy as np
from typing import Union

# Constants
AUDIO_EXTENSIONS = ("wav", "mp3", "ogg", "flac")
AUDIO_INPUT_DIR = "audio"

# Functions
def fullpath(filepath: str) -> str:
    """
    Returns a full filepath for the given filepath.
    """
    return os.path.abspath(os.path.expanduser(filepath))

def audioInputDir() -> str:
    """
    Returns the audio input directory.
    """
    return os.path.join(os.path.dirname(__file__), AUDIO_INPUT_DIR)

def normalizeArray(
        array: Union[np.ndarray, torch.Tensor],
        minVal: float = 0.0,
        maxVal: float = 1.0
    ) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes the given array between minVal and maxVal.

    array: A Numpy array or a Tensor.
    minVal: The minimum value of the normalized array.
    maxVal: The maximum value of the normalized array.

    Returns a normalized Numpy array or Tensor matching the `array` type.
    """
    arrayMin = torch.min(array) if isinstance(array, torch.Tensor) else np.min(array)
    arrayMax = torch.max(array) if isinstance(array, torch.Tensor) else np.max(array)
    return minVal + (array - arrayMin) * (maxVal - minVal) / (arrayMax - arrayMin)
