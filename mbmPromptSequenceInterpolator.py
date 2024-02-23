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
            promptSeq: Optional[InterpPromptSequence] = None
            relDesiredFrames = math.ceil(desiredFrames / (promptCount - 1))
            for i in range(promptCount - 1):
                # Calculate modifiers for this section
                curModifiers = feat_mods[(relDesiredFrames * i):(relDesiredFrames * (i + 1))]

                # Build prompt interpolation
                if promptSeq is None:
                    # Start intial prompt sequence
                    promptSeq = InterpPromptSequence(prompts[i], prompts[i + 1], curModifiers)
                else:
                    # Expand prompt sequence
                    promptSeq.addToSequence(prompts[i], prompts[i + 1], curModifiers)

            # Trim off any extra frames produced from ceil to int
            promptSeq.trimToLength(desiredFrames)

            # Set the initial prompt
            promptPos = MbmPrompt.buildComfyUiPrompt(
                promptSeq.positives[0],
                promptSeq.positivePools[0]
            )
            promptNeg = MbmPrompt.buildComfyUiPrompt(
                promptSeq.negatives[0],
                promptSeq.negativePools[0]
            )
        elif promptCount == 1:
            # Set single prompt
            promptPos = prompts[0].positivePrompt()
            promptNeg = prompts[0].negativePrompt()
        else:
            # No prompts
            raise ValueError("No prompts were provided to the Music Visualizer node. At least one prompt is required.")

        return (
            # TODO: prompt sequence
            # TODO: charts
        )

