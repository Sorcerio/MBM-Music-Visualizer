# MBM's Music Visualizer: The Visualizer
# Visualize a provided audio file.

# TODO: Feature: Add filebased input (json) for prompt sequence.
# TODO: Feature: Add ability to specify specific timecodes for prompts.
# TODO: Feature: Add ability to use a hash (?) of a latent (image?) to generate a dummy (random?) audio input.
# TODO: Feature: Add camera effects similar to Scene Weaver.
# TODO: Feature: Add ability to drag in audio files to the loader.

# Imports
import librosa
import torch
import random
import math
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from tqdm import tqdm
from scipy.signal import resample
from PIL import Image

import comfy.samplers
from nodes import common_ksampler

from .mbmPromptSequence import MbmPrompt

# Classes
class MusicVisualizer:
    """
    Visualize a provided audio file.

    Returns a batch tuple of images.
    """
    # Class Constants
    SEED_MODE_FIXED = "fixed"
    SEED_MODE_RANDOM = "random"
    SEED_MODE_INCREASE = "increase"
    SEED_MODE_DECREASE = "decrease"

    LATENT_MODE_STATIC = "static"
    LATENT_MODE_INCREASE = "increase"
    LATENT_MODE_DECREASE = "decrease"
    LATENT_MODE_FLOW = "flow"
    LATENT_MODE_GAUSS = "guassian"
    LATENT_MODE_BOUNCE = "bounce"

    FEAT_APPLY_METHOD_ADD = "add"
    FEAT_APPLY_METHOD_SUBTRACT = "subtract"

    DEF_FEAT_MOD_MAX = 10000.0
    DEF_FEAT_MOD_MIN = -10000.0

    RETURN_TYPES = ("LATENT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("LATENTS", "FPS", "CHARTS")
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Audio"

    # Constructor
    def __init__(self):
        self.__isBouncingUp = True # Used when `bounce` mode is used to track direction

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "prompts": ("PROMPT_SEQ", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_image": ("LATENT", ),
                "seed_mode": ([MusicVisualizer.SEED_MODE_FIXED, MusicVisualizer.SEED_MODE_RANDOM, MusicVisualizer.SEED_MODE_INCREASE, MusicVisualizer.SEED_MODE_DECREASE], ),
                "latent_mode": ([MusicVisualizer.LATENT_MODE_BOUNCE, MusicVisualizer.LATENT_MODE_FLOW, MusicVisualizer.LATENT_MODE_STATIC, MusicVisualizer.LATENT_MODE_INCREASE, MusicVisualizer.LATENT_MODE_DECREASE, MusicVisualizer.LATENT_MODE_GAUSS], ),
                "intensity": ("FLOAT", {"default": 1.0}), # Muiltiplier for the audio features
                "hop_length": ("INT", {"default": 512}),
                "fps_target": ("FLOAT", {"default": 6, "min": -1, "max": 10000}), # Provide `<= 0` to use whatever audio sampling comes up with
                "image_limit": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}), # Provide `<= 0` to use whatever audio sampling comes up with
                "latent_mod_limit": ("FLOAT", {"default": 5.0, "min": -1.0, "max": 10000.0}), # The maximum variation that can occur to the latent based on the latent's mean value. Provide `<= 0` to have no limit
                "feat_mod_max": ("FLOAT", {"default": s.DEF_FEAT_MOD_MAX, "min": -10000.0, "max": 10000.0}), # The maximum value the feature modifier can be. 10,000 should be unattainable through normal usage.
                "feat_mod_min": ("FLOAT", {"default": s.DEF_FEAT_MOD_MIN, "min": -10000.0, "max": 10000.0}), # The minimum value the feature modifier can be. -10,000 should be unattainable through normal usage.
                "feat_mod_normalize": ([False, True], ), # If `True`, the feature modifier array will be normalized between 0 and the maximum value in the array.

                # TODO: Move these into a KSamplerSettings node?
                # Also might be worth adding a KSamplerSettings to KSamplerInputs node that splits it all out to go into the standard KSampler when done here?
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self,
            audio: tuple,
            prompts: list[MbmPrompt],
            seed: int,
            latent_image: dict[str, torch.Tensor],
            seed_mode: str,
            latent_mode: str,
            intensity: float,
            hop_length: int,
            fps_target: float,
            image_limit: int,
            latent_mod_limit: float,
            feat_mod_max: float,
            feat_mod_min: float,
            feat_mod_normalize: bool,

            model,
            steps: int,
            cfg: float,
            sampler_name: str,
            scheduler: str,
            denoise: float,
        ):
        ## Validation
        # Make sure if bounce mode is used that the latent mod limit is set
        if (latent_mode == MusicVisualizer.LATENT_MODE_BOUNCE) and (latent_mod_limit <= 0):
            raise ValueError("Latent Mod Limit must be set to `>0` when using the `bounce` Latent Mode")

        # Make sure the feature modifier values are valid
        if feat_mod_max < feat_mod_min:
            raise ValueError("The maximum feature modifier value must be greater than the minimum feature modifier value.")

        if feat_mod_max == self.DEF_FEAT_MOD_MAX:
            feat_mod_max = None

        if feat_mod_min == self.DEF_FEAT_MOD_MIN:
            feat_mod_min = None

        ## Setup Calculations
        # Set the random library seeds
        random.seed(seed)
        np.random.default_rng(seed)

        # Unpack the audio
        y, sr = audio

        # Calculate the duration of the audio
        duration = librosa.get_duration(y=y, sr=sr, hop_length=hop_length)
        # hopSeconds = hop_length / sr

        # Calculate tempo
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = self._normalizeArray(librosa.beat.tempo(onset_envelope=onset, sr=sr, hop_length=hop_length, aggregate=None))

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

        # Normalize the spectro mean
        spectroMean = self._normalizeArray(spectroMean)

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
        chromaMean = resample(chromaMean, desiredFrames)

        # Calculate the feature modifier for each frame
        featModifiers = torch.Tensor(
            [self._calcFeatModifier(
                intensity,
                tempo[i],
                spectroMean[i],
                chromaMean[i],
                modMax=feat_mod_max,
                modMin=feat_mod_min
            ) for i in range(desiredFrames)]
        )

        # Normalize the feature modifiers if requested
        if feat_mod_normalize:
            # NOTE: The feat_mod_max and feat_mod_min values will be inaccurate realtive to the now normalized featModifiers array and must be adjusted if used later.
            featModifiers = self._normalizeArray(featModifiers, minVal=0.0, maxVal=featModifiers.max().item())

        ## Generation
        # Set intial prompts
        promptCount = len(prompts)
        if promptCount > 1:
            # Calculate linear interpolation between prompts
            promptSeqPos = None
            promptSeqNeg = None
            promptSeqPosPool = None
            promptSeqNegPool = None
            relDesiredFrames = math.ceil(desiredFrames / (promptCount - 1))
            for i in range(promptCount - 1):
                curModifiers = featModifiers[(relDesiredFrames * i):(relDesiredFrames * (i + 1))]
                if promptSeqPos is None: # Keep it in Tensor space for efficiency.
                    promptSeqPos = self._weightedInterpolation(
                        prompts[i].positive,
                        prompts[i + 1].positive,
                        curModifiers
                    )
                    promptSeqNeg = self._weightedInterpolation(
                        prompts[i].negative,
                        prompts[i + 1].negative,
                        curModifiers
                    )
                    promptSeqPosPool = self._weightedInterpolation(
                        prompts[i].positivePool,
                        prompts[i + 1].positivePool,
                        curModifiers
                    )
                    promptSeqNegPool = self._weightedInterpolation(
                        prompts[i].negativePool,
                        prompts[i + 1].negativePool,
                        curModifiers
                    )
                else:
                    promptSeqPos = torch.vstack((
                        promptSeqPos,
                        self._weightedInterpolation(
                            prompts[i].positive,
                            prompts[i + 1].positive,
                            curModifiers
                        )[1:]
                    ))
                    promptSeqNeg = torch.vstack((
                        promptSeqNeg,
                        self._weightedInterpolation(
                            prompts[i].negative,
                            prompts[i + 1].negative,
                            curModifiers
                        )[1:]
                    ))
                    promptSeqPosPool = torch.vstack((
                        promptSeqPosPool,
                        self._weightedInterpolation(
                            prompts[i].positivePool,
                            prompts[i + 1].positivePool,
                            curModifiers
                        )[1:]
                    ))
                    promptSeqNegPool = torch.vstack((
                        promptSeqNegPool,
                        self._weightedInterpolation(
                            prompts[i].negativePool,
                            prompts[i + 1].negativePool,
                            curModifiers
                        )[1:]
                    ))

            # Trim off any extra frames produced from ceil to int
            promptSeqPos = promptSeqPos[:desiredFrames]
            promptSeqNeg = promptSeqNeg[:desiredFrames]
            promptSeqPosPool = promptSeqPosPool[:desiredFrames]
            promptSeqNegPool = promptSeqNegPool[:desiredFrames]

            # Set the initials prompt
            promptPos = MbmPrompt.buildComfyUiPrompt(promptSeqPos[0], promptSeqPosPool[0])
            promptNeg = MbmPrompt.buildComfyUiPrompt(promptSeqNeg[0], promptSeqNegPool[0])
        elif promptCount == 1:
            # Set single prompt
            promptPos = prompts[0].positivePrompt()
            promptNeg = prompts[0].negativePrompt()
        else:
            # No prompts
            raise ValueError("No prompts were provided to the Music Visualizer node. At least one prompt is required.")

        # Prepare latent output tensor
        outputTensor: torch.Tensor = None
        latentTensorMeans = np.zeros(desiredFrames)
        latentTensor = latent_image["samples"].clone()
        for i in (pbar := tqdm(range(desiredFrames), desc="Music Visualization")):
            # Calculate the latent tensor
            latentTensor = self._iterateLatentByMode(
                latentTensor,
                latent_mode,
                latent_mod_limit,
                featModifiers[i]
            )

            # Records the latent tensor's mean
            latentTensorMeans[i] = torch.mean(latentTensor).numpy()

            # Set progress bar info
            pbar.set_postfix({
                "feat": f"{featModifiers[i]:.2f}",
                "prompt": f"{torch.mean(promptPos[0][0]):.4f}",
                "latent": f"{latentTensorMeans[i]:.2f}"
            })

            # Generate the image
            imgTensor = common_ksampler(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    promptPos,
                    promptNeg,
                    {"samples": latentTensor}, # ComfyUI, why package it?
                    denoise=denoise
                )[0]["samples"]

            if outputTensor is None:
                outputTensor = imgTensor
            else:
                outputTensor = torch.vstack((
                    outputTensor,
                    # latentTensor,
                    imgTensor
                ))

            # Limit if one if supplied
            if (image_limit > 0) and (i >= (image_limit - 1)):
                break

            # Iterate seed as needed
            seed = self._iterateSeedByMode(seed, seed_mode)

            # Iterate the prompts as needed
            if (promptCount > 1) and ((i + 1) < desiredFrames):
                promptPos = MbmPrompt.buildComfyUiPrompt(promptSeqPos[i + 1], promptSeqPosPool[i + 1])
                promptNeg = MbmPrompt.buildComfyUiPrompt(promptSeqNeg[i + 1], promptSeqNegPool[i + 1])

        # Render charts
        chartImages = torch.vstack([
            self._chartGenerationFeats(
                {
                    "seed": f"{seed} ({seed_mode})",
                    "latent mode": latent_mode,
                    "latent mod limit": f"{latent_mod_limit:.2f}",
                    "intensity": f"{intensity:.2f}",
                    "feat mod max": (f"{feat_mod_max:.2f}" if (feat_mod_max is not None) else "none"),
                    "feat mod min": (f"{feat_mod_min:.2f}" if (feat_mod_min is not None) else "none"),
                    "feat mod norm": ("yes" if feat_mod_normalize else "no"),
                    "hop length": hop_length,
                    "fps target": f"{fps_target:.2f}",
                    "frames": desiredFrames
                },
                tempo,
                spectroMean,
                chromaMean,
                featModifiers,
                promptSeqPos
            ),
            self._chartData(tempo, "Tempo"),
            self._chartData(spectroMean, "Spectro Mean"),
            self._chartData(chromaMean, "Chroma Mean"),
            self._chartFeatMod(featModifiers, feat_mod_normalize, modMax=feat_mod_max, modMin=feat_mod_min),
            self._chartData(latentTensorMeans, "Latent Means"),
            self._chartData([torch.mean(c) for c in promptSeqPos], "Positive Prompt"),
            self._chartData([torch.mean(c) for c in promptSeqNeg], "Negative Prompt")
        ])

        # Return outputs
        return ({"samples": outputTensor}, fps, chartImages)

    # Private Functions
    def _normalizeArray(self,
            array: Union[np.ndarray, torch.Tensor],
            minVal: float = 0.0,
            maxVal: float = 1.0
        ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalizes the given array between minVal and maxVal.

        array: A Numpy array or a Tensor.
        minVal: The minimum value of the normalized array.
        maxVal: The maximum value of the normalized array.

        Returns a normalized Numpy array or Tensor matching the `array` type.
        """
        arrayMin = torch.min(array) if isinstance(array, torch.Tensor) else np.min(array)
        arrayMax = torch.max(array) if isinstance(array, torch.Tensor) else np.max(array)
        return minVal + (array - arrayMin) * (maxVal - minVal) / (arrayMax - arrayMin)

    def _iterateLatentByMode(self,
            latent: torch.Tensor,
            latentMode: str,
            modLimit: float,
            modifier: float
        ) -> torch.Tensor:
        """
        Produces a latent tensor based on the provided mode.

        latent: The latent tensor to modify.
        latentMode: The mode to iterate by.
        modLimit: The maximum variation that can occur to the latent based on the latent's mean value. Provide `<= 0` to have no limit.
        modifier: The amount to modify the latent by each hop.

        Returns the iterated latent tensor.
        """
        # Decide what to do if in flow mode
        if latentMode == MusicVisualizer.LATENT_MODE_FLOW:
            # Each hop will add or subtract, based on the audio features, from the last latent
            if random.random() < 0.5:
                latentMode = MusicVisualizer.LATENT_MODE_INCREASE
            else:
                latentMode = MusicVisualizer.LATENT_MODE_DECREASE

        if latentMode == MusicVisualizer.LATENT_MODE_BOUNCE:
            # Increases to to the `modLimit`, then decreases to `-modLimit`, and loops as many times as needed building on the last latent
            if modLimit > 0:
                # Do the bounce operation
                # Calculate the next value
                curLatentMean = torch.mean(latent)
                nextValue = (curLatentMean + modifier) if self.__isBouncingUp else (curLatentMean - modifier)

                # Check if within bounds
                if -modLimit <= nextValue <= modLimit:
                    # Within bounds
                    latentMode = MusicVisualizer.LATENT_MODE_INCREASE if self.__isBouncingUp else MusicVisualizer.LATENT_MODE_DECREASE
                else:
                    # Outside of bounds
                    latentMode = MusicVisualizer.LATENT_MODE_DECREASE if self.__isBouncingUp else MusicVisualizer.LATENT_MODE_INCREASE
                    self.__isBouncingUp = not self.__isBouncingUp
            else:
                # No limit so just increase
                latentMode = MusicVisualizer.LATENT_MODE_INCREASE

        # Decide what to do based on mode
        if latentMode == MusicVisualizer.LATENT_MODE_INCREASE:
            # Each hop adds, based on the audio features, to the last latent
            return self._applyFeatToLatent(latent, MusicVisualizer.FEAT_APPLY_METHOD_ADD, modLimit, modifier)
        elif latentMode == MusicVisualizer.LATENT_MODE_DECREASE:
            # Each hop subtracts, based on the audio features, from the last latent
            return self._applyFeatToLatent(latent, MusicVisualizer.FEAT_APPLY_METHOD_SUBTRACT, modLimit, modifier)
        elif latentMode == MusicVisualizer.LATENT_MODE_GAUSS:
            # Each hop creates a new latent with guassian noise
            return self._createLatent(latent.shape)
        else: # LATENT_MODE_STATIC
            # Only the provided latent is used ignoring audio features
            return latent

    def _createLatent(self, size: tuple) -> torch.Tensor:
        """
        Creates a latent tensor from normal distribution noise.

        size: The size of the latent tensor.

        Returns the latent tensor.
        """
        # TODO: More specific noise range input
        return torch.tensor(np.random.normal(3, 2.5, size=size))

    def _calcFeatModifier(self,
            intensity: float,
            tempo: float,
            spectroMean: float,
            chromaMean: float,
            modMax: Optional[float] = None,
            modMin: Optional[float] = None
        ) -> float:
        """
        Calculates the overall feature modifier based on the provided audio features.

        intensity: A modifier to increase (>1.0) or decrease (<1.0) the overall effect of the audio features.
        tempo: The tempo for a single step of the audio.
        spectroMean: The normalized mean power for a single step of the audio.
        chromaMean: The mean value of the chroma for a single step of the audio.
        modMax: The maximum value the feature modifier can be. Provide `None` to have no maximum.
        modMin: The minimum value the feature modifier can be. Provide `None` to have no minimum.

        Returns the calculated overall feature modifier.
        """
        modVal = (((tempo + 1.0) * (spectroMean + 1.0) * (chromaMean + 1.0)) * intensity)

        if (modMax is not None) and (modVal > modMax):
            return modMax
        elif (modMin is not None) and (modVal < modMin):
            return modMin
        else:
            return modVal

    def _applyFeatToLatent(self,
            latent: torch.Tensor,
            method: str,
            modLimit: float,
            modifier: float
        ) -> torch.Tensor:
        """
        Applys the provided features to the latent tensor.

        latent: The latent tensor to modify.
        method: The method to use to apply the features.
        modLimit: The maximum variation that can occur to the latent based on the latent's mean value. Provide `<= 0` to have no limit.
        modifier: The amount to modify the latent by each hop.

        Returns the modified latent tensor.
        """
        # Apply features to every point in the latent
        if method == MusicVisualizer.FEAT_APPLY_METHOD_ADD:
            # Add the features to the latent
            # Check if mean will be exceeded
            if (modLimit > 0) and (torch.mean(latent) + modifier) > modLimit:
                # Mean is exceeded so latent only
                return latent

            # Add the features to the latent
            latent += modifier
        else: # FEAT_APPLY_METHOD_SUBTRACT
            # Subtract the features from the latent
            # Check if mean will be exceeded
            if (modLimit > 0) and (torch.mean(latent) - modifier) < -modLimit:
                # Mean is exceeded so latent only
                return latent

            # Subtract the features from the latent
            latent -= modifier

        return latent

    def _iterateSeedByMode(self, seed: int, seedMode: str):
        """
        Produces a seed based on the provided mode.

        seed: The seed to iterate.
        seedMode: The mode to iterate by.

        Returns the iterated seed.
        """
        if seedMode == MusicVisualizer.SEED_MODE_RANDOM:
            # Seed is random every hop
            return random.randint(0, 0xffffffffffffffff)
        elif seedMode == MusicVisualizer.SEED_MODE_INCREASE:
            # Seed increases by 1 every hop
            return seed + 1
        elif seedMode == MusicVisualizer.SEED_MODE_DECREASE:
            # Seed decreases by 1 every hop
            return seed - 1
        else: # SEED_MODE_FIXED
            # Seed stays the same
            return seed

    def _weightedInterpolation(self, start: torch.Tensor, stop: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Interpolates between `start` and `stop` based on the given `weights` for each step in the interpolation.

        start: A Tensor of the same shape as `stop`.
        stop: A Tensor of the same shape as `start`.
        weights: A Tensor of weights to to use in each jump of the interpolation. This defines the number of gaps, not the total number of output Tensor elements.

        Returns a Tensor of shape `[(length of weights + 1), *start.shape]` where each step is an interpolation between `start` and `stop`.
        Includes the `start` or `stop` tensors in the output.
        """
        # Make sure weights are floats
        weights = weights.float()

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate the cumulative sum of the weights
        cumWeight = weights.cumsum(dim=0)

        # Reshape the cumulative sum to allow for broadcasting
        for _ in range(start.ndim): # [-1, *([1]*start.ndim)]
            cumWeight = cumWeight.unsqueeze(-1)

        # Interpolate with weights and add start
        return torch.vstack([start.unsqueeze(0), (start[None] + cumWeight * (stop - start)[None])])

    def _renderChart(self, fig: plt.Figure) -> torch.Tensor:
        """
        Renders the provided chart.

        fig: The chart to render.

        Returns a ComfyUI compatible Tensor image of the chart.
        """
        # Render the chart
        fig.canvas.draw()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)

        # Convert to an image tensor
        return torch.from_numpy(
            np.array(
                Image.open(buffer).convert("RGB")
            ).astype(np.float32) / 255.0
        )[None,]

    def _chartData(self, data: Union[np.ndarray, torch.Tensor], title: str, dotValues: bool = False) -> torch.Tensor:
        """
        Creates a chart of the provided data.

        data: A numpy array or a Tensor to chart.
        title: The title of the chart.
        dotValues: If data points should be added as dots on top of the line.

        Returns a ComfyUI compatible Tensor image of the chart.
        """
        # Build the chart
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(data)
        ax.grid(True)

        if dotValues:
            ax.scatter(range(len(data)), data, color="red")

        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

        # Render the chart
        return self._renderChart(fig)

    def _chartGenerationFeats(self,
            renderParams: dict[str, str],
            tempo,
            spectroMean,
            chromaMean,
            featModifiers,
            promptSeqPos
        ) -> torch.Tensor:
        """
        Creates a chart representing the entire generation flow.

        renderParams: The parameters used to render the chart.
        tempo: The tempo feature data.
        spectroMean: The spectrogram mean feature data.
        featModifiers: The calculated feature modifiers.
        promptSeqPos: The positive prompt sequence.

        Returns a ComfyUI compatible Tensor image of the chart.
        """
        # Build the chart
        fig, ax = plt.subplots(figsize=(20, 4))

        ax.plot(self._normalizeArray(tempo), label="Tempo") 
        ax.plot(self._normalizeArray(spectroMean), label="Spectro Mean")
        ax.plot(self._normalizeArray(chromaMean), label="Chroma Mean")
        ax.plot(self._normalizeArray(featModifiers), label="Modifiers")
        ax.plot(self._normalizeArray([torch.mean(c) for c in promptSeqPos]), label="Prompt")
    
        ax.legend()
        ax.grid(True)
        ax.set_title("Combined")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

        # Add the render parameters
        renderParams = "\n".join([f"{str(k).strip()}: {str(v).strip()}" for k, v in renderParams.items()])
        ax.text(1.02, 0.5, renderParams, transform=ax.transAxes, va="center")

        # Render the chart
        return self._renderChart(fig)

    def _chartFeatMod(self,
            featModifiers: torch.Tensor,
            isNormalized: bool,
            modMax: Optional[float] = None,
            modMin: Optional[float] = None
        ) -> torch.Tensor:
        """
        Creates a chart representing the feature modifier.

        featModifiers: The calculated feature modifiers.
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
        featureInfo += f"Min: {modMinStr}"
        ax.text(1.02, 0.5, featureInfo, transform=ax.transAxes, va="center")

        # Render the chart
        return self._renderChart(fig)
