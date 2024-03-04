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
    def __init__(self):
        pass

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

    RETURN_TYPES = ("PROMPT_SEQ", )
    RETURN_NAMES = ("PROMPTS", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Prompts"

    def process(self, filepath: str, clip) -> list[MbmPrompt]:
        # Load the prompt sequence data
        with open(os.path.join(promptSeqInputDir(), filepath), "r") as file:
            promptData = json.load(file)

        # Check if prompts are present
        promptKey = "prompts"
        if promptKey not in promptData:
            raise ValueError("No prompts found in the Prompt Sequence file.")

        # Create the text encoder
        textEncoder = CLIPTextEncode()

        # Loop through the data
        promptsOut: list[MbmPrompt] = []
        promptSet: dict[str, Any]
        for promptSet in promptData[promptKey]:
            # Create the packaged prompt
            curPrompt = MbmPrompt.fromComfyUiPrompts(
                textEncoder.encode(clip=clip, text=promptSet["positive"])[0],
                textEncoder.encode(clip=clip, text=promptSet["negative"])[0]
            )

            # Add additional data if present
            PromptSequenceData.tryToAddDataFromJson(curPrompt, promptSet)

            # Record the prompt
            promptsOut.append(curPrompt)

        return (promptsOut, )
