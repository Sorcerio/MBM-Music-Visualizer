# MBM's Music Visualizer

# Imports
from mbmMusicVisualizer import MusicVisualizer

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    # "id": Class
    "mbmMusicVisuzalizer": MusicVisualizer
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # "id": "readable"
    "mbmMusicVisuzalizer": "Music Visualizer"
}

# Export
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
