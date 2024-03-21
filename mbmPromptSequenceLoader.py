# MBM's Music Visualizer: Prompt Sequence Loader
# Loads a Prompt Sequence from a JSON input file.

# Imports
import os
import json
from typing import Any

from nodes import CLIPTextEncode

from .mbmMVShared import PROMPT_SEQ_EXTENSIONS, promptSeqInputDir
from .mbmPrompt import MbmPrompt
from .mbmPromptSequenceData import PromptSequenceData

# Classes
class PromptSequenceLoader:
    """
    Loads a Prompt Sequence from a JSON input file.
    """
    # Class Constants
    KEY_DEFAULTS = "defaults"
    KEY_SEQUENCE = "sequence"
    KEY_POSITIVE = "positive"
    KEY_NEGATIVE = "negative"
    KEY_TIMECODE = "timecode"

    RETURN_TYPES = ("PROMPT_SEQ", )
    RETURN_NAMES = ("PROMPTS", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Prompts"

    # Constructor
    def __init__(self):
        pass

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        inputDir = promptSeqInputDir()
        localFiles = [f for f in os.listdir(inputDir) if (os.path.isfile(os.path.join(inputDir, f)) and (os.path.splitext(f)[1].strip(".").lower() in PROMPT_SEQ_EXTENSIONS))]

        return {
            "required": {
                "filepath": (sorted(localFiles), ),
                "clip": ("CLIP", )
            }
        }

    def process(self, filepath: str, clip) -> list[MbmPrompt]:
        # Load the prompt sequence data
        jsonPath = os.path.join(promptSeqInputDir(), filepath)
        with open(jsonPath, "r") as file:
            promptData = json.load(file)

        # Validate the JSON
        self.validateJson(promptData)

        # Check any prompts were provided
        if len(promptData[self.KEY_SEQUENCE]) == 0:
            # Report and return
            print(f"No prompts provided in the `sequence` for: {jsonPath}")
            return ([], )

        # Create the text encoder
        textEncoder = CLIPTextEncode()

        # Loop through the data
        promptsOut: list[MbmPrompt] = []
        promptSet: dict[str, Any]
        for promptSet in promptData[self.KEY_SEQUENCE]:
            # Get the current prompts
            curPosPrompt = promptSet.get(
                self.KEY_POSITIVE,
                promptData[self.KEY_DEFAULTS][self.KEY_POSITIVE]
            )
            curNegPrompt = promptSet.get(
                self.KEY_NEGATIVE,
                promptData[self.KEY_DEFAULTS][self.KEY_NEGATIVE]
            )

            # Create the packaged prompt
            curPrompt = MbmPrompt.fromComfyUiPrompts(
                textEncoder.encode(clip=clip, text=curPosPrompt)[0],
                textEncoder.encode(clip=clip, text=curNegPrompt)[0]
            )

            # Add additional data if present
            PromptSequenceData.tryToAddDataFromJson(curPrompt, promptSet)

            # Record the prompt
            promptsOut.append(curPrompt)

        return (promptsOut, )

    # Functions
    def validateJson(self, json):
        """
        Validates the given JSON data or throws a `ValueError`.
        """
        # Validate the JSON
        if self.KEY_DEFAULTS not in json:
            raise ValueError("No `defaults` provided in the Prompt Sequence file.")
        else:
            # Check that the defaults have the required keys
            if self.KEY_POSITIVE not in json[self.KEY_DEFAULTS]:
                raise ValueError("No `positive` prompt provided in the defaults of the Prompt Sequence file.")

            if self.KEY_NEGATIVE not in json[self.KEY_DEFAULTS]:
                raise ValueError("No `negative` prompt provided in the defaults of the Prompt Sequence file.")

        if self.KEY_SEQUENCE not in json:
            raise ValueError("No `sequence` provided in the Prompt Sequence file.")
