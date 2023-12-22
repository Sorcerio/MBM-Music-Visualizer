# MBM's Music Visualizer: Shared
# Shared functionality for the Music Visualizer.

# Imports
import os

# Constants
AUDIO_EXTENSIONS = ("wav", "mp3", "ogg", "flac")
AUDIO_INPUT_DIR = "audio"

# Functions
def fullpath(filepath: str) -> str:
    """
    Returns a full filepath for the given filepath.
    """
    return os.path.abspath(os.path.expanduser(filepath))

def audioInputDir() -> str:
    """
    Returns the audio input directory.
    """
    return os.path.join(os.path.dirname(__file__), AUDIO_INPUT_DIR)
