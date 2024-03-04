# MBM's Music Visualizer: Prompt Sequence
# Allows for building of a sequence of prompts.

# Imports
import torch
from typing import Union

from .mbmPrompt import MbmPrompt
from .mbmPromptSequenceData import PromptSequenceData

# Classes
class PromptSequenceBuilder:
    """
    Allows for building of a sequence of prompts.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", )
            },
            "optional": {
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "prompts": ("PROMPT_SEQ", )
            }
        }

    RETURN_TYPES = ("PROMPT_SEQ", )
    RETURN_NAMES = ("PROMPTS", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Prompts"

    def process(self,
            positive_1: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]],
            negative_1: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]],
            positive_2: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]] = None,
            negative_2: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]] = None,
            prompts: list[MbmPrompt] = None
        ) -> list[MbmPrompt]:
        """
        Returns a list of MbmPrompt objects representing the prompt sequence.
        """
        # Create a new prompt sequence
        if (positive_2 is None) or (negative_2 is None):
            # Include just the required one
            promptsOut = [
                MbmPrompt.fromComfyUiPrompts(positive_1, negative_1)
            ]
        else:
            # Include both
            promptsOut = [
                MbmPrompt.fromComfyUiPrompts(positive_1, negative_1),
                MbmPrompt.fromComfyUiPrompts(positive_2, negative_2)
            ]

        # Add to the given prompt sequence if provided
        if prompts is not None:
            promptsOut = (prompts + promptsOut)

        return (promptsOut, )

class PromptSequenceBuilderAdvanced(PromptSequenceBuilder):
    """
    Allows for building prompts with additional information required for more fine control of prompt sequences.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputTypes = super().INPUT_TYPES()

        inputTypes["required"]["timecode_1"] = ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0xffffffffffffffff})
        inputTypes["optional"]["timecode_2"] = ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0xffffffffffffffff})

        return inputTypes

    def process(self,
            positive_1: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]],
            negative_1: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]],
            timecode_1: float,
            positive_2: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]] = None,
            negative_2: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]] = None,
            timecode_2: float = 0.0,
            prompts: list[MbmPrompt] = None
        ):
        """
        Returns a list of MbmPrompt objects representing the prompt sequence.
        """
        # Validation
        if (timecode_1 < 0) or (timecode_2 < 0):
            raise ValueError("Timecodes must be >= 0.0.")

        # Run the super
        promptsOut = super().process(positive_1, negative_1, positive_2, negative_2, prompts)

        # Add the sequence data
        if (positive_2 is None):
            # Add data to only the last prompt
            promptsOut[0][-1].data[PromptSequenceData.DATA_KEY] = PromptSequenceData(timecode_1)
        else:
            # Add data to the proper prompts
            promptsOut[0][-2].data[PromptSequenceData.DATA_KEY] = PromptSequenceData(timecode_1)
            promptsOut[0][-1].data[PromptSequenceData.DATA_KEY] = PromptSequenceData(timecode_2)

        return promptsOut
