# MBM's Music Visualizer: Audio Loader
# Load an audio file.

# Imports
import os
import mbmMVShared as mvs

# Classes
class MusicVisualizer:
    """
    Load an audio file.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputDir = mvs.audioInputDir()
        localFiles = [f for f in os.listdir(inputDir) if os.path.isfile(os.path.join(inputDir, f))]
        return {
            "required": {
                "file": (sorted(localFiles), {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes"

    def process(self):
        return None