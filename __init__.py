# MBM's Music Visualizer

# Imports
from .mbmMusicVisualizer import MusicVisualizer
from .mbmAudioLoader import AudioLoader
from .mbmPromptSequence import PromptSequenceBuilder

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    # "id": Class
    "mbmMusicVisuzalizer": MusicVisualizer,
    "mbmAudioLoader": AudioLoader,
    "mbmPromptSequenceBuilder": PromptSequenceBuilder
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # "id": "readable"
    "mbmMusicVisuzalizer": "Music Visualizer",
    "mbmAudioLoader": "Audio Loader",
    "mbmPromptSequenceBuilder": "Prompt Sequence Builder"
}

# Export
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
