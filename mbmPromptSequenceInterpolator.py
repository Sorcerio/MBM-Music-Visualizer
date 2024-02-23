# MBM's Music Visualizer: Prompt Sequence Interpolator
# Interpolates additional prompts from a prompt sequence based on input.

# Imports
import torch
import math
from typing import Optional

from .mbmPromptSequence import MbmPrompt
from .mbmInterpPromptSequence import InterpPromptSequence
from .mbmMVShared import chartData

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

        # Render the charts
        chartImages = torch.vstack([
            chartData([torch.mean(pt.positive) for pt in promptSeq], "Positive Prompt"),
            chartData([torch.mean(pt.negative) for pt in promptSeq], "Negative Prompt")
        ])

        return (
            promptSeq,
            chartImages
        )

