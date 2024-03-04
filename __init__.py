# MBM's Music Visualizer

# Imports
from .mbmAudioFeatureCalculator import AudioFeatureCalculator
from .mbmAudioLoader import AudioLoader
from .mbmPromptSequenceRenderer import PromptSequenceRenderer
from .mbmPromptSequenceBuilder import PromptSequenceBuilder, PromptSequenceBuilderAdvanced
from .mbmPromptSequenceInterpolator import PromptSequenceInterpolator
from .mbmImageConcat import ImageConcatenator
from .mbmPromptSequenceLoader import PromptSequenceLoader

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    # "id": Class
    "mbmAudioFeatureCalculator": AudioFeatureCalculator,
    "mbmAudioLoader": AudioLoader,
    "mbmPromptSequenceRenderer": PromptSequenceRenderer,
    "mbmPromptSequenceBuilder": PromptSequenceBuilder,
    "mbmPromptSequenceBuilderAdv": PromptSequenceBuilderAdvanced,
    "mbmPromptSequenceInterpolator": PromptSequenceInterpolator,
    "mbmImageConcat": ImageConcatenator,
    "mbmPromptSequenceLoader": PromptSequenceLoader
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # "id": "readable"
    "mbmAudioFeatureCalculator": "Audio Feature Calculator",
    "mbmAudioLoader": "Audio Loader",
    "mbmPromptSequenceRenderer": "Prompt Sequence Renderer",
    "mbmPromptSequenceBuilder": "Prompt Sequence Builder",
    "mbmPromptSequenceBuilderAdv": "Prompt Sequence Builder (Advanced)",
    "mbmPromptSequenceInterpolator": "Prompt Sequence Interpolator",
    "mbmImageConcat": "Image Concatenator",
    "mbmPromptSequenceLoader": "Prompt Sequence Loader"
}

# Export
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
