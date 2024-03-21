# MBM's Music Visualizer: Prompt Sequence Interpolator
# Interpolates additional prompts from a prompt sequence based on input.

# Imports
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional

from .mbmPrompt import MbmPrompt
from .mbmPromptSequenceData import PromptSequenceData
from .mbmInterpPromptSequence import InterpPromptSequence
from .mbmMVShared import chartData, renderChart, normalizeArray

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

        # Prepare the prompt distribution chart
        promptChart = self._startPromptChart()

        # Get the lengths
        promptCount = len(prompts)
        desiredFrames = len(feat_mods)

        # Set intial prompts
        if promptCount > 1:
            # Calculate linear interpolation between prompts
            interpPromptSeq: Optional[InterpPromptSequence] = None
            relDesiredFrames = math.ceil(desiredFrames / (promptCount - 1))
            interPromptStartIndex = 0
            for i in range(promptCount - 1):
                # Get the relevant prompts
                curPrompt = prompts[i]
                nextPrompt = prompts[i + 1]

                # Decide on modifiers calculation
                if split_mode == self.INTERP_OP_TIMECODE:
                    # Calculate based on timecode
                    # Get the timecodes
                    startTimecode = PromptSequenceData.getDataFromPrompt(curPrompt).timecode
                    endTimecode = PromptSequenceData.getDataFromPrompt(nextPrompt).timecode

                    # Check for first and final prompt
                    if i == 0:
                        # First prompt. Start at 0
                        startTimecode = 0
                    elif i == (promptCount - 2):
                        # Last prompt. Go to the end
                        endTimecode = (desiredFrames * feat_seconds)

                    # Calculate modifiers
                    curModifiers = self._selectFeaturesWithTimecode(feat_mods, feat_seconds, startTimecode, endTimecode)
                else: # INTERP_OP_EVEN
                    # Calculate modifiers for this section with even distribution
                    curModifiers = feat_mods[(relDesiredFrames * i):(relDesiredFrames * (i + 1))]

                # Build prompt interpolation
                if interpPromptSeq is None:
                    # Start intial prompt sequence
                    interpPromptSeq = InterpPromptSequence(curPrompt, nextPrompt, curModifiers)
                else:
                    # Iterate the iter index
                    interPromptStartIndex = len(interpPromptSeq.positives)

                    # Expand prompt sequence
                    interpPromptSeq.addToSequence(curPrompt, nextPrompt, curModifiers)

                # Add the prompt identifier line
                self._addPromptIndicator(promptChart, i, (interPromptStartIndex - 1), interPromptStartIndex)

                if i == (promptCount - 2):
                    # Mark the final prompt
                    self._addPromptIndicator(promptChart, (i + 1), desiredFrames, len(interpPromptSeq.positives))

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

        # Plot the prompt distribution
        promptChart[1].plot(list(range(len(promptSeq))), normalizeArray([torch.mean(pt.positive) for pt in promptSeq]), label="Positive")
        promptChart[1].plot(list(range(len(promptSeq))), normalizeArray([torch.mean(pt.negative) for pt in promptSeq]), label="Negative")
        promptChart[1].plot(list(range(len(promptSeq))), normalizeArray(feat_mods), label="Feature Modifiers", linestyle="dotted")

        # Render the charts
        chartImages = torch.vstack([
            chartData([torch.mean(pt.positive) for pt in promptSeq], "Positive Prompt"),
            chartData([torch.mean(pt.negative) for pt in promptSeq], "Negative Prompt"),
            self._renderPromptChart(promptChart)
        ])

        return (
            promptSeq,
            chartImages
        )

    # Private Functions
    def _selectFeaturesWithTimecode(self,
            featMods: torch.Tensor,
            featSeconds: float,
            startTimecode: float,
            endTimecode: float,
        ) -> torch.Tensor:
        """
        Selects the proper features based on the timecode of the current and next prompts.

        featMods: The feature modifiers tensor.
        featSeconds: The number of seconds each item in the `featMods` represents.
        startTimecode: The timecode of the current prompt.
        endTimecode: The timecode of the next prompt.

        Returns a trimmed Tensor based on the `featMods` and the timecodes of the prompts.
        """
        # Calculate the features
        return featMods[math.floor(startTimecode / featSeconds):math.ceil(endTimecode / featSeconds)]

    def _startPromptChart(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Handles initial creation of the prompt distribution chart.
        """
        # Build the chart
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.grid(True)
        ax.set_prop_cycle(color=list(mcolors.TABLEAU_COLORS.values()))

        return (fig, ax)

    def _addPromptIndicator(self, chart: tuple[plt.Figure, plt.Axes], index: int, frameNum: int, toFrameNum: int):
        """
        Adds a prompt indicator to the given chart.

        chart: The chart to add the indicator to.
        index: The index of the prompt.
        frameNum: The frame number to add the indicator at.
        toFrameNum: The frame number the prompt is replaced by the next prompt.
        """
        chart[1].axvline(x=frameNum, linestyle="dashed", color="red")
        chart[1].text(frameNum + 0.5, 0.5, f"Prompt {index + 1} ({toFrameNum})", rotation=90, verticalalignment="center")

    def _renderPromptChart(self, chart: tuple[plt.Figure, plt.Axes]) -> torch.Tensor:
        """
        Renders the prompt distribution chart.
        """
        # Add labels
        chart[1].set_title("Normalized Prompt Distribution")
        chart[1].set_xlabel("Frame Number")
        chart[1].set_ylabel("Prompt Mean (Normalized)")
        chart[1].legend()

        # Render the chart
        return renderChart(chart[0])
