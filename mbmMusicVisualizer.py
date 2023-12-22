# MBM's Music Visualizer: The Visualizer
# Visualize a provided audio file.

# Imports
# etc

# Classes
class MusicVisualizer:
    """
    Visualize a provided audio file.

    Returns a batch tuple of images.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": "AUDIO" # TODO: add other params
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGES", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes"

    def process(self, audio):
        return None