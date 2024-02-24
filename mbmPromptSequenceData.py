# MBM's Music Visualizer: MBM Prompt, Sequence Data
# Data class for representing data relevant for advanced prompt sequences to include in an `MbmPrompt` object's `data` dictionary.

# Imports
from .mbmPrompt import MbmPrompt

# Classes
class PromptSequenceData:
    """
    Data class for representing data relevant for advanced prompt sequences to include in an `MbmPrompt` object's `data` dictionary.
    """
    # Class Constants
    DATA_KEY = "sequenceData"

    # Constructor
    def __init__(self,
            timecode: float
        ) -> None:
        """
        timecode: A timecode in seconds reprsenting the time when the prompt should be fully present in the sequence.
        """
        self.timecode = timecode

    # Python Functions
    def __repr__(self) -> str:
        return f"PromptSequenceData({', '.join(self.__dict__)})"

    # Static Functions
    @staticmethod
    def promptHasSequenceData(prompt: MbmPrompt) -> bool:
        """
        Returns if the given prompt has sequence data.
        """
        return (PromptSequenceData.DATA_KEY in prompt.data)
    
    @staticmethod
    def getDataFromPrompt(prompt: MbmPrompt) -> 'PromptSequenceData':
        """
        Returns the sequence data from the given prompt if present.
        Throws a `ValueError` if the data is not present.
        """
        # Get the data or fail
        if PromptSequenceData.promptHasSequenceData(prompt):
            return prompt.data[PromptSequenceData.DATA_KEY]
        else:
            raise ValueError(f"The given prompt does not have Sequence Data attached to it. Issue prompt: {prompt}")
