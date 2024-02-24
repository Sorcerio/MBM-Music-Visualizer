# MBM's Music Visualizer: Prompt Sequence Interpolator
# Interpolates additional prompts from a prompt sequence based on input.

# Imports
import torch
import math
from typing import Optional

from .mbmPrompt import MbmPrompt
from .mbmPromptSequenceData import PromptSequenceData
from .mbmInterpPromptSequence import InterpPromptSequence
from .mbmMVShared import chartData

# Classes
class PromptSequenceInterpolator:
    """
    Interpolates additional prompts from a prompt sequence based on input.
    """
    # Class Constants
    INTERP_OP_EVEN = "split_evenly"
    INTERP_OP_TIMECODE = "split_on_timecode"

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
                "feat_mods": ("TENSOR_1D", ),
                "split_mode": ([s.INTERP_OP_EVEN, s.INTERP_OP_TIMECODE], )
            },
            "optional": {
                "feat_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0}) # The number of seconds each item in the `feat_mods` represents. If <=0, prompts will be evenly split across the feature modifiers.
            }
        }

    def process(self,
            prompts: list[MbmPrompt],
            feat_mods: torch.Tensor,
            split_mode: str,
            feat_seconds: float = -1.0
        ):
        # Validation
        if (split_mode == self.INTERP_OP_TIMECODE) and (feat_seconds <= 0):
            raise ValueError(f"The feature item duration (in seconds; the amount of time each item in the `feat_mods` represents) must be provided when using `{self.INTERP_OP_TIMECODE}` mode.")

        # Get the lengths
        promptCount = len(prompts)
        desiredFrames = len(feat_mods)

        # Set intial prompts
        if promptCount > 1:
            # Calculate linear interpolation between prompts
            interpPromptSeq: Optional[InterpPromptSequence] = None
            relDesiredFrames = math.ceil(desiredFrames / (promptCount - 1))
            for i in range(promptCount - 1):
                # Get the relevant prompts
                curPrompt = prompts[i]
                nextPrompt = prompts[i + 1]

                # Decide on modifiers calculation
                if split_mode == self.INTERP_OP_TIMECODE:
                    # Calculate based on timecode
                    curModifiers = self._selectFeaturesWithTimecode(feat_mods, curPrompt, nextPrompt, feat_seconds)
                else: # INTERP_OP_EVEN
                    # Calculate modifiers for this section with even distribution
                    curModifiers = feat_mods[(relDesiredFrames * i):(relDesiredFrames * (i + 1))]

                # Build prompt interpolation
                if interpPromptSeq is None:
                    # Start intial prompt sequence
                    interpPromptSeq = InterpPromptSequence(curPrompt, nextPrompt, curModifiers)
                else:
                    # Expand prompt sequence
                    interpPromptSeq.addToSequence(curPrompt, nextPrompt, curModifiers)

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

    def _selectFeaturesWithTimecode(self,
            featMods: torch.Tensor,
            featSeconds: float,
            curPrompt: MbmPrompt,
            nextPrompt: MbmPrompt,
        ) -> torch.Tensor:
        """
        Selects the proper features based on the timecode of the current and next prompts.

        featMods: The feature modifiers tensor.
        featSeconds: The number of seconds each item in the `featMods` represents.
        curPrompt: The current prompt.
        nextPrompt: The next prompt.

        Returns a trimmed Tensor based on the `featMods` and the timecodes of the prompts.
        """
        # Get the timecodes
        startTimecode = PromptSequenceData.getDataFromPrompt(curPrompt).timecode
        endTimecode = PromptSequenceData.getDataFromPrompt(nextPrompt).timecode

        # Calculate the features
        return featMods[int(startTimecode / featSeconds):int(endTimecode / featSeconds)]
