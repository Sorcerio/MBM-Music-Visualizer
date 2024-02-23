# MBM's Music Visualizer: Interpolated Prompt Sequence
# Represents and facilitates building of a sequence of interpolated prompts.

# Imports
import torch

from .mbmPromptSequence import MbmPrompt

# Classes
class InterpPromptSequence:
    """
    Represents and facilitates building of a sequence of interpolated prompts.
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
        )
        self.negatives = self._weightedInterpolation(
            start.negative,
            end.negative,
            modifiers
        )
        self.positivePools = self._weightedInterpolation(
            start.positivePool,
            end.positivePool,
            modifiers
        )
        self.negativePools = self._weightedInterpolation(
            start.negativePool,
            end.negativePool,
            modifiers
        )

    # Functions
    def addToSequence(self, start: MbmPrompt, end: MbmPrompt, modifiers: torch.Tensor) -> None:
        """
        Add additional interpolated prompts to the sequence.

        start: The prompt to start from.
        end: The prompt to end on.
        modifiers: The feature modifiers to use in the weighted interpolation.
        """
        self.positives = torch.vstack((
            self.positives,
            self._weightedInterpolation(
                start.positive,
                end.positive,
                modifiers
            )[1:]
        ))
        self.negatives = torch.vstack((
            self.negatives,
            self._weightedInterpolation(
                start.negative,
                end.negative,
                modifiers
            )[1:]
        ))
        self.positivePools = torch.vstack((
            self.positivePools,
            self._weightedInterpolation(
                start.positivePool,
                end.positivePool,
                modifiers
            )[1:]
        ))
        self.negativePools = torch.vstack((
            self.negativePools,
            self._weightedInterpolation(
                start.negativePool,
                end.negativePool,
                modifiers
            )[1:]
        ))

    def trimToLength(self, length: int) -> None:
        """
        Trims the sequence to the provided length.

        length: The length to trim the sequence to.
        """
        self.positives = self.positives[:length]
        self.negatives = self.negatives[:length]
        self.positivePools = self.positivePools[:length]
        self.negativePools = self.negativePools[:length]

    def _weightedInterpolation(self, start: torch.Tensor, stop: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Interpolates between `start` and `stop` based on the given `weights` for each step in the interpolation.

        start: A Tensor of the same shape as `stop`.
        stop: A Tensor of the same shape as `start`.
        weights: A Tensor of weights to to use in each jump of the interpolation. This defines the number of gaps, not the total number of output Tensor elements.

        Returns a Tensor of shape `[(length of weights + 1), *start.shape]` where each step is an interpolation between `start` and `stop`.
        Includes the `start` or `stop` tensors in the output.
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

        # Interpolate with weights and add start
        return torch.vstack([start.unsqueeze(0), (start[None] + cumWeight * (stop - start)[None])])
