# MBM's Music Visualizer: Shared
# Shared functionality for the Music Visualizer.

# Imports
import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from PIL import Image

# Constants
AUDIO_EXTENSIONS = ("wav", "mp3", "ogg", "flac")
AUDIO_INPUT_DIR = "audio"

PROMPT_SEQ_EXTENSIONS = ("json", )
PROMPT_SEQ_INPUT_DIR = "promptSequences"

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

def promptSeqInputDir() -> str:
    """
    Returns the prompt sequence input directory.
    """
    return os.path.join(os.path.dirname(__file__), PROMPT_SEQ_INPUT_DIR)

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

def renderChart(fig: plt.Figure) -> torch.Tensor:
    """
    Renders the provided chart.

    fig: The chart to render.

    Returns a ComfyUI compatible Tensor image of the chart.
    """
    # Render the chart
    fig.canvas.draw()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    # Convert to an image tensor
    return torch.from_numpy(
        np.array(
            Image.open(buffer).convert("RGB")
        ).astype(np.float32) / 255.0
    )[None,]

def chartData(data: Union[np.ndarray, torch.Tensor], title: str, dotValues: bool = False) -> torch.Tensor:
    """
    Creates a chart of the provided data.

    data: A numpy array or a Tensor to chart.
    title: The title of the chart.
    dotValues: If data points should be added as dots on top of the line.

    Returns a ComfyUI compatible Tensor image of the chart.
    """
    # Build the chart
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(data)
    ax.grid(True)

    if dotValues:
        ax.scatter(range(len(data)), data, color="red")

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")

    # Render the chart
    return renderChart(fig)
