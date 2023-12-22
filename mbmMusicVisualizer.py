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
                "audio": "AUDIO",
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "pitch": ("INT", {"default": 220}), # sensitivity
                "tempo": ("INT", {"default": 220}), # sensitivity
                "depth": ("FLOAT", {"default": 1.0}),
                "jitter": ("FLOAT", {"default": 0.5}),
                "truncation": ("FLOAT", {"default": 1.0}),
                "smoothing": ("INT", {"default": 20}), # factor
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGES", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes"

    def process(self,
        audio,
        width: int,
        height: int,
        pitch: int,
        tempo: int,
        depth: float,
        jitter: float,
        truncation: float,
        smoothing: int
    ):
        return None