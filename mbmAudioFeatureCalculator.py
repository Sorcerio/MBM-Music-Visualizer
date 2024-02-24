# MBM's Music Visualizer: Audio Feature Calculator
# Calculates relevant audio features from loaded audio.

# Imports
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.signal import resample

from .mbmMVShared import normalizeArray, chartData, renderChart

# Classes
class AudioFeatureCalculator:
    """
    Calculates relevant audio features from loaded audio.
    """
    # Class Constants
    DEF_FEAT_MOD_MAX = 10000.0
    DEF_FEAT_MOD_MIN = -10000.0

    RETURN_TYPES = ("TENSOR_1D", "FLOAT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("FEAT_MODS", "FEAT_SECONDS", "FPS", "CHARTS")
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Audio"

    # Constructor
    def __init__(self):
        pass

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "intensity": ("FLOAT", {"default": 1.0}), # Muiltiplier for the audio features
                "hop_length": ("INT", {"default": 512}),
                "fps_target": ("FLOAT", {"default": 6, "min": -1, "max": 10000}), # Provide `<= 0` to use whatever audio sampling comes up with
                "feat_mod_max": ("FLOAT", {"default": s.DEF_FEAT_MOD_MAX, "min": -10000.0, "max": 10000.0}), # The maximum value the feature modifier can be. 10,000 should be unattainable through normal usage.
                "feat_mod_min": ("FLOAT", {"default": s.DEF_FEAT_MOD_MIN, "min": -10000.0, "max": 10000.0}), # The minimum value the feature modifier can be. -10,000 should be unattainable through normal usage.
                "feat_mod_normalize": ([False, True], ), # If `True`, the feature modifier array will be normalized between 0 and the maximum value in the array.
            }
        }

    def process(self,
            audio: tuple,
            intensity: float,
            hop_length: int,
            fps_target: float,
            feat_mod_max: float,
            feat_mod_min: float,
            feat_mod_normalize: bool
        ):
        ## Validation
        # Make sure the feature modifier values are valid
        if feat_mod_max < feat_mod_min:
            raise ValueError("The maximum feature modifier value must be greater than the minimum feature modifier value.")

        if feat_mod_max == self.DEF_FEAT_MOD_MAX:
            feat_mod_max = None

        if feat_mod_min == self.DEF_FEAT_MOD_MIN:
            feat_mod_min = None

        ## Calculation
        # Unpack the audio
        y, sr = audio

        # Calculate the duration of the audio
        duration = librosa.get_duration(y=y, sr=sr, hop_length=hop_length)

        # # Calculate the duration of a single hop
        # hopSeconds: float = (hop_length / sr)

        # Calculate tempo
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = normalizeArray(librosa.beat.tempo(onset_envelope=onset, sr=sr, hop_length=hop_length, aggregate=None))

        # Calculate the spectrogram
        spectro = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=8000,
            hop_length=hop_length
        )

        # Calculate normalized mean power per hop
        spectroMean = np.mean(spectro, axis=0)

        # Calculate the delta of the spectrogram mean
        spectroMeanDelta = librosa.feature.delta(spectroMean)

        # Normalize the spectro mean
        spectroMean = normalizeArray(spectroMean)

        # Calculate pitch chroma for hops
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=hop_length
        )

        # Get the mean of the chroma for each step
        chromaMean = np.mean(chroma, axis=0)

        # Calculate the output FPS
        if fps_target <= 0:
            # Calculate framerate based on audio
            fps = len(tempo) / duration
        else:
            # Specific framerate to target
            fps = fps_target

        # Calculate desired frame count
        desiredFrames = round(fps * duration)

        # Resample audio features to match desired frame count
        tempo = resample(tempo, desiredFrames)
        spectroMean = resample(spectroMean, desiredFrames)
        spectroMeanDelta = resample(spectroMeanDelta, desiredFrames)
        chromaMean = resample(chromaMean, desiredFrames)

        # Calculate the feature modifier for each frame
        featModifiers = torch.Tensor(
            [self._calcFeatModifier(
                intensity,
                tempo[i],
                spectroMean[i],
                spectroMeanDelta[i],
                chromaMean[i],
                modMax=feat_mod_max,
                modMin=feat_mod_min
            ) for i in range(desiredFrames)]
        )

        # Normalize the feature modifiers if requested
        if feat_mod_normalize:
            # NOTE: The feat_mod_max and feat_mod_min values will be inaccurate realtive to the now normalized featModifiers array and must be adjusted if used later.
            featModifiers = normalizeArray(featModifiers, minVal=0.0, maxVal=featModifiers.max().item())

        # Build the charts
        chartImages = torch.vstack([
            chartData(tempo, "Tempo"),
            chartData(spectroMean, "Spectrogram Mean"),
            chartData(spectroMeanDelta, "Spectrogram Mean Delta"),
            chartData(chromaMean, "Chroma Mean"),
            self._chartFeatMod(featModifiers, intensity, feat_mod_normalize, modMax=feat_mod_max, modMin=feat_mod_min)
        ])

        return (
            featModifiers,
            (duration / desiredFrames),
            fps,
            chartImages
        )
    
    # Internal Functions
    def _calcFeatModifier(self,
            intensity: float,
            tempo: float,
            spectroMean: float,
            spectroMeanDelta: float,
            chromaMean: float,
            modMax: Optional[float] = None,
            modMin: Optional[float] = None
        ) -> float:
        """
        Calculates the overall feature modifier based on the provided audio features.

        intensity: A modifier to increase (>1.0) or decrease (<1.0) the overall effect of the audio features.
        tempo: The tempo for a single step of the audio.
        spectroMean: The normalized mean power for a single step of the audio.
        spectroMeanDelta: The delta of the normalized mean power for a single step of the audio.
        chromaMean: The mean value of the chroma for a single step of the audio.
        modMax: The maximum value the feature modifier can be. Provide `None` to have no maximum.
        modMin: The minimum value the feature modifier can be. Provide `None` to have no minimum.

        Returns the calculated overall feature modifier.
        """
        modVal = (((tempo + 1.0) * (spectroMean + 1.0) * (chromaMean + 1.0)) * (intensity + spectroMeanDelta))

        if (modMax is not None) and (modVal > modMax):
            return modMax
        elif (modMin is not None) and (modVal < modMin):
            return modMin
        else:
            return modVal

    def _chartFeatMod(self,
            featModifiers: torch.Tensor,
            intensity: float,
            isNormalized: bool,
            modMax: Optional[float] = None,
            modMin: Optional[float] = None
        ) -> torch.Tensor:
        """
        Creates a chart representing the feature modifier.

        featModifiers: The calculated feature modifiers.
        intensity: The given intensity used in calculating the feature modifiers.
        isNormalized: If the feature modifiers are normalized.
        modMax: The maximum value the feature modifier can be. Provide `None` to display no maximum.
        modMin: The minimum value the feature modifier can be. Provide `None` to display no minimum.

        Returns a ComfyUI compatible Tensor image of the chart.
        """
        # Build the chart
        fig, ax = plt.subplots(figsize=(20, 4))

        ax.plot(featModifiers, label="Modifiers")

        modMaxStr = "none"
        if modMax is not None:
            if isNormalized:
                # Use logical value
                modMaxAlt = featModifiers.max().item()
                modMaxStr = f"{modMaxAlt:.2f} ({modMax:.2f})"
                plt.axhline(y=modMaxAlt, linestyle="dotted", color="yellow", label="Mod Max")
            else:
                # Use prescribed value
                modMaxStr = f"{modMax:.2f}"
                plt.axhline(y=modMax, linestyle="dotted", color="yellow", label="Mod Max")

        modMinStr = "none"
        if modMin is not None:
            if isNormalized:
                # Use logical value
                modMinStr = f"0 ({modMin:.2f})"
                plt.axhline(y=0, linestyle="dotted", color="green", label="Mod Min")
            else:
                # Use prescribed value
                modMinStr = f"{modMin:.2f}"
                plt.axhline(y=modMin, linestyle="dotted", color="green", label="Mod Min")

        ax.legend()
        ax.grid(True)
        ax.set_title("Feature Modifiers")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

        # Add feature information
        featureInfo = f"Normalized: {('yes' if isNormalized else 'no')}\n"
        featureInfo += f"Max: {modMaxStr}\n"
        featureInfo += f"Min: {modMinStr}\n"
        featureInfo += f"Intensity: {intensity:.2f}"
        ax.text(1.02, 0.5, featureInfo, transform=ax.transAxes, va="center")

        # Render the chart
        return renderChart(fig)
