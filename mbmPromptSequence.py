# MBM's Music Visualizer: Prompt Sequence
# Allows for building of a sequence of prompts.

# Imports
import torch
from typing import Union

from .mbmPrompt import MbmPrompt

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
        ):
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
