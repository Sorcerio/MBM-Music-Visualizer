# MBM's Music Visualizer: Prompt Sequence
# Allows for building of a sequence of prompts.

# Imports
import torch
from typing import Optional, Union

# Classes
class MbmPrompt:
    """
    Easier to work with Prompt data container class since ComfyUI just stores them unannotated _lists_ and embedded dictionaries.
    """
    # Constructor
    def __init__(self,
            positive: torch.Tensor,
            negative: torch.Tensor,
            positivePool: Optional[torch.Tensor],
            negativePool: Optional[torch.Tensor]
        ) -> None:
        """
        positive: The positive prompt tensor.
        negative: The negative prompt tensor.
        positivePool: The positive prompt's pooled output tensor. If `None` is provided, SDXL models _will not_ function.
        negativePool: The negative prompt's pooled output tensor. If `None` is provided, SDXL models _will not_ function.
        """
        self.positive = positive
        self.negative = negative
        self.positivePool = positivePool
        self.negativePool = negativePool

        # TODO: Add additional "data" parameter for additional data to be stored with the prompt

    # Python Functions
    def __repr__(self) -> str:
        return f"MbmPrompt(positive={self.positive.shape}, negative={self.negative.shape}, positivePool={self.positivePool.shape if self.positivePool is not None else 'None'}, negativePool={self.negativePool.shape if self.negativePool is not None else 'None'})"

    # Class Functions
    @classmethod
    def fromComfyUiPrompts(cls, positive: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]], negative: list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]]) -> 'MbmPrompt':
        """
        Constructs a new MbmPrompt object from the given ComfyUI compatible prompts.
        """
        return cls(
            positive[0][0].squeeze(),
            negative[0][0].squeeze(),
            positivePool=positive[0][1]["pooled_output"].squeeze() if ("pooled_output" in positive[0][1]) else None,
            negativePool=negative[0][1]["pooled_output"].squeeze() if ("pooled_output" in negative[0][1]) else None
        )

    # Static Functions
    @staticmethod
    def buildComfyUiPrompt(prompt: torch.Tensor, pool: Optional[torch.Tensor] = None) -> list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]]:
        """
        Returns the given prompt data as a ComfyUI compatible prompt.
        """
        return [[prompt.unsqueeze(0), {"pooled_output": pool.unsqueeze(0)}]]

    # Functions
    def positivePrompt(self) -> list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]]:
        """
        Returns the positive prompt as a ComfyUI compatible prompt.
        """
        return MbmPrompt.buildComfyUiPrompt(self.positive, self.positivePool)

    def negativePrompt(self) -> list[list[Union[torch.Tensor, dict[str, torch.Tensor]]]]:
        """
        Returns the negative prompt as a ComfyUI compatible prompt.
        """
        return MbmPrompt.buildComfyUiPrompt(self.negative, self.negativePool)

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
