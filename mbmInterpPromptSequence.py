# MBM's Music Visualizer: Interpolated Prompt Sequence
# Represents and facilitates building of a sequence of prompts with weighted interpolation.

# Imports
import torch

from .mbmPrompt import MbmPrompt

# Classes
class InterpPromptSequence:
    """
    Represents and facilitates building of a sequence of prompts with weighted interpolation.
    """
    # Constructor
    def __init__(self, start: MbmPrompt, end: MbmPrompt, modifiers: torch.Tensor) -> None:
        """
        start: The prompt to start from.
        end: The prompt to end on.
        modifiers: The feature modifiers to use in the weighted interpolation.
        """
        # Calculate initial interpolation
        self.positives = self._weightedInterpolation(
            start.positive,
            end.positive,
            modifiers
        )[0]
        self.negatives = self._weightedInterpolation(
            start.negative,
            end.negative,
            modifiers
        )[0]
        self.positivePools = self._weightedInterpolation(
            start.positivePool,
            end.positivePool,
            modifiers
        )[0]
        self.negativePools = self._weightedInterpolation(
            start.negativePool,
            end.negativePool,
            modifiers
        )[0]

    # Functions
    def addToSequence(self, start: MbmPrompt, end: MbmPrompt, modifiers: torch.Tensor) -> None:
        """
        Add additional interpolated prompts to the sequence.

        start: The prompt to start from.
        end: The prompt to end on.
        modifiers: The feature modifiers to use in the weighted interpolation.
        """
        # Add each data type to the sequence
        self.positives = self._addToSequence(self.positives, start.positive, end.positive, modifiers)
        self.negatives = self._addToSequence(self.negatives, start.negative, end.negative, modifiers)
        self.positivePools = self._addToSequence(self.positivePools, start.positivePool, end.positivePool, modifiers)
        self.negativePools = self._addToSequence(self.negativePools, start.negativePool, end.negativePool, modifiers)

    def _addToSequence(self, container: torch.Tensor, start: torch.Tensor, stop: torch.Tensor, modifiers: torch.Tensor) -> torch.Tensor:
        """
        Add an additional interpolated tensor to the sequence.

        container: The tensor to add the interpolation to.
        start: The tensor to start from.
        stop: The tensor to end on.
        modifiers: The feature modifiers to use in the weighted interpolation.

        Returns the modified `container` tensor.
        """
        # Calculate the interpolation
        interp, shapeChanged = self._weightedInterpolation(
            start,
            stop,
            modifiers,
            tokenCount=container.size(1)
        )
        interp = interp[1:]

        # Check if shape updates are needed
        if shapeChanged:
            container = self.addPromptTokens(container, interp.shape)

        # Create modified container with the interpolation
        return torch.vstack((container, interp))

    def trimToLength(self, length: int) -> None:
        """
        Trims the sequence to the provided length.

        length: The length to trim the sequence to.
        """
        self.positives = self.positives[:length]
        self.negatives = self.negatives[:length]
        self.positivePools = self.positivePools[:length]
        self.negativePools = self.negativePools[:length]

    def asPromptSequence(self) -> list[MbmPrompt]:
        """
        Returns the sequence as a list of MbmPrompt objects.
        """
        return [
            MbmPrompt(
                self.positives[i],
                self.negatives[i],
                positivePool=self.positivePools[i],
                negativePool=self.negativePools[i]
            )
            for i in range(len(self.positives))
        ]

    def _weightedInterpolation(self, start: torch.Tensor, stop: torch.Tensor, weights: torch.Tensor, tokenCount: int = -1) -> tuple[torch.Tensor, bool]:
        """
        Interpolates between `start` and `stop` based on the given `weights` for each step in the interpolation.

        start: A Tensor of the same shape as `stop`.
        stop: A Tensor of the same shape as `start`.
        weights: A Tensor of weights to to use in each jump of the interpolation. This defines the number of gaps, not the total number of output Tensor elements.
        tokenCount: The number of tokens in the prompt tensor. If the number of tokens in the prompt tensor is less than this value, zero-padding will be added to match the length.

        Returns a tuple of:
        * A Tensor of shape `[(length of weights + 1), *start.shape]` where each step is an interpolation between `start` and `stop`. Includes the `start` or `stop` tensors in the output.
        * A boolean indicating if the shape of the output Tensor has changed.
        """
        # Make sure weights are floats
        weights = weights.float()

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate the cumulative sum of the weights
        cumWeight = weights.cumsum(dim=0)

        # Reshape the cumulative sum to allow for broadcasting
        for _ in range(start.ndim): # [-1, *([1]*start.ndim)]
            cumWeight = cumWeight.unsqueeze(-1)

        # Resample the shorter tensor if needed
        shapeChanged = False
        maxLen = max(start.size(0), stop.size(0), tokenCount)

        if start.size(0) < maxLen:
            shapeChanged = True
            start = torch.vstack([start, torch.zeros(maxLen - start.size(0), start.size(1))])

        if stop.size(0) < maxLen:
            shapeChanged = True
            stop = torch.vstack([stop, torch.zeros(maxLen - stop.size(0), stop.size(1))])

        # Interpolate with weights and add start
        return (
            torch.vstack([start.unsqueeze(0), (start[None] + cumWeight * (stop - start)[None])]),
            shapeChanged
        )

    def addPromptTokens(self, tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Adds zero-padding to the given prompt tensor to match the provided `shape` along the dimension that represents the number of tokens in the prompt tensor.

        tensor: The prompt tensor to add zero-padding to.
        shape: The shape to match the prompt tensor to.

        Returns the modified tensor or the original tensor if no padding is needed.
        """
        # Calculate the size of the zero-padding needed
        paddingSize = shape[1] - tensor.shape[1]

        # Check if padding is needed
        if paddingSize <= 0:
            return tensor

        # Concatenate the original tensor and the zero-padding along dimension 1
        return torch.cat(
            [
                tensor,
                torch.zeros(tensor.shape[0], paddingSize, tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
            ],
            dim=1
        )
