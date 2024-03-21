# MBM's Music Visualizer: Audio Loader
# Load an audio file.

# Imports
import os
import librosa

from .mbmMVShared import AUDIO_EXTENSIONS, fullpath, audioInputDir

# Classes
class AudioLoader:
    """
    Load an audio file.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputDir = audioInputDir()
        localFiles = [f for f in os.listdir(inputDir) if (os.path.isfile(os.path.join(inputDir, f)) and (os.path.splitext(f)[1].strip(".").lower() in AUDIO_EXTENSIONS))]

        return {
            "required": {
                "filepath": (sorted(localFiles), )
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "FILENAME")
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Audio"

    def process(self, filepath: str):
        filepath = os.path.join(audioInputDir(), filepath)

        return (
            librosa.load(fullpath(filepath)),
            os.path.splitext(os.path.basename(filepath))[0]
        )