# MBM's Music Visualizer: Prompt Sequence Interpolator
# Interpolates additional prompts from a prompt sequence based on input.

# Imports
import librosa
import torch
import random
import math
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from tqdm import tqdm
from scipy.signal import resample
from PIL import Image

import comfy.samplers
from nodes import common_ksampler

from .mbmPromptSequence import MbmPrompt
from .mbmInterpPromptSequence import InterpPromptSequence

# Classes
class PromptSequenceInterpolator:
    """
    Interpolates additional prompts from a prompt sequence based on input.
    """
    # Class Constants
    RETURN_TYPES = ("PROMPT_SEQ", "IMAGE")
    RETURN_NAMES = ("PROMPTS", "CHARTS")
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Prompts"

    # Constructor
    def __init__(self):
        pass

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("PROMPT_SEQ", ),
                "feat_mods": ("TENSOR_1D", )
            }
        }

    def process(self,
            prompts: list[MbmPrompt],
            feat_mods: torch.Tensor
        ):
        # Get the lengths
        promptCount = len(prompts)
        desiredFrames = len(feat_mods)

        # Set intial prompts
        if promptCount > 1:
            # Calculate linear interpolation between prompts
            interpPromptSeq: Optional[InterpPromptSequence] = None
            relDesiredFrames = math.ceil(desiredFrames / (promptCount - 1))
            for i in range(promptCount - 1):
                # Calculate modifiers for this section
                curModifiers = feat_mods[(relDesiredFrames * i):(relDesiredFrames * (i + 1))]

                # Build prompt interpolation
                if interpPromptSeq is None:
                    # Start intial prompt sequence
                    interpPromptSeq = InterpPromptSequence(prompts[i], prompts[i + 1], curModifiers)
                else:
                    # Expand prompt sequence
                    interpPromptSeq.addToSequence(prompts[i], prompts[i + 1], curModifiers)

            # Trim off any extra frames produced from ceil to int
            interpPromptSeq.trimToLength(desiredFrames)

            # Build the prompt sequence
            promptSeq = interpPromptSeq.asPromptSequence()
        elif promptCount == 1:
            # Send it onward
            promptSeq = prompts
        else:
            # No prompts my guy
            raise ValueError("At least one prompt is required.")

        return (
            promptSeq,
            # TODO: charts
        )

