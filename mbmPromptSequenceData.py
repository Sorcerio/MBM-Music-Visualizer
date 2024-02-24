# MBM's Music Visualizer: MBM Prompt, Sequence Data
# Data class for representing data relevant for advanced prompt sequences to include in an `MbmPrompt` object's `data` dictionary.

# Classes
class MbmPromptSequenceData:
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
        return f"MbmPromptSequenceData({', '.join(self.__dict__)})"
