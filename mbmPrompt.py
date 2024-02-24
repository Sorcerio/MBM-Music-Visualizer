# MBM's Music Visualizer: MBM Prompt
# Easier to work with Prompt data container class since ComfyUI just stores them in unannotated lists and embedded dictionaries.

# Imports
import torch
from typing import Optional, Union

# Classes
class MbmPrompt:
    """
    Easier to work with Prompt data container class since ComfyUI just stores them in unannotated lists and embedded dictionaries.
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
        self.data = {}

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
