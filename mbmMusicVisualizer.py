# MBM's Music Visualizer: The Visualizer
# Visualize a provided audio file.

# Imports
import librosa
import numpy as np

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
                "audio": ("AUDIO",),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "pitch": ("INT", {"default": 220}), # sensitivity
                "tempo": ("INT", {"default": 220}), # sensitivity
                "depth": ("FLOAT", {"default": 1.0}),
                "jitter": ("FLOAT", {"default": 0.5}),
                "truncation": ("FLOAT", {"default": 1.0}),
                "smoothing": ("INT", {"default": 20}), # factor
                "hopLength": ("INT", {"default": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGES", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/MusicVisualizer"

    def process(self,
        audio: tuple,
        width: int,
        height: int,
        pitchSensitivity: int,
        tempoSensitivity: int,
        depth: float,
        jitter: float,
        truncation: float,
        smoothFactor: int,
        hopLen: int
    ):
        # Unpack the audio
        y, sr = audio

        # Calculate parameters
        pitchSensitivity = (300 - pitchSensitivity) * 512 / hopLen
        tempoSensitivity = tempoSensitivity * hopLen / 512

        if smoothFactor > 1:
            smoothFactor = int(smoothFactor * 512 / hopLen)

        # Calculate the spectrogram
        spectro = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=8000,
            hop_length=hopLen
        )

        # Calculate mean power for hops
        spectroMean = np.mean(spectro, axis=0)

        # Calculate power gradient for hops
        spectroGrad = np.gradient(spectroMean)
        spectroGrad = (spectroGrad / np.max(spectroGrad)).clip(min=0)

        # Normalize mean power
        spectroMean = (spectroMean / np.min(spectroMean)) / np.ptp(spectroMean)

        # Calculate pitch chroma for hops
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=hopLen
        )

        # Sort pitch chroma
        chromaSort = np.argsort(np.mean(chroma, axis=1))[::-1]
