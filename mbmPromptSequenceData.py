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
            timecode: float,
            **kwargs
        ) -> None:
        """
        timecode: A timecode in seconds reprsenting the time when the prompt should be fully present in the sequence.
        """
        self.timecode = float(timecode)

    # Python Functions
    def __repr__(self) -> str:
        contents = [f"{k} = {v}" for k, v in self.__dict__.items()]
        return f"PromptSequenceData({', '.join(contents)})"

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

    @staticmethod
    def addDataToPrompt(prompt: MbmPrompt, data: 'PromptSequenceData'):
        """
        Adds the given data to the prompt.
        """
        prompt.data[PromptSequenceData.DATA_KEY] = data

    @staticmethod
    def tryToAddDataFromJson(prompt: MbmPrompt, jsonData: dict) -> bool:
        """
        Tries to add sequence data from the given JSON data to the prompt.

        prompt: The prompt to add the data to.
        jsonData: The JSON data to map to a `PromptSequenceData` object constructor.

        Returns `True` if the operation was successful.
        """
        # Collapse if needed
        if PromptSequenceData.DATA_KEY in jsonData:
            jsonData = jsonData[PromptSequenceData.DATA_KEY]

        # Try to get the data
        try:
            data = PromptSequenceData(**jsonData)
        except NameError:
            # The data is not present
            return False
        except TypeError:
            # No timecode supplied
            return False

        # Add the data
        PromptSequenceData.addDataToPrompt(prompt, data)

        return True
