# MBM's Music Visualizer: The Visualizer
# Visualize a provided audio file.

# Imports
import librosa
import torch
import random
import numpy as np
from tqdm import tqdm

import comfy.samplers
from nodes import common_ksampler

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

    FEAT_APPLY_METHOD_ADD = "add"
    FEAT_APPLY_METHOD_SUBTRACT = "subtract"

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("LATENTS", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/MusicVisualizer"

    # Constructor
    def __init__(self):
        pass

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # TODO: EVERYTHING must use the seed
                "latent_image": ("LATENT", ),
                "seed_mode": ([MusicVisualizer.SEED_MODE_FIXED, MusicVisualizer.SEED_MODE_RANDOM, MusicVisualizer.SEED_MODE_INCREASE, MusicVisualizer.SEED_MODE_DECREASE], ),
                "latent_mode": ([MusicVisualizer.LATENT_MODE_FLOW, MusicVisualizer.LATENT_MODE_STATIC, MusicVisualizer.LATENT_MODE_INCREASE, MusicVisualizer.LATENT_MODE_DECREASE, MusicVisualizer.LATENT_MODE_GAUSS], ),
                "pitch": ("INT", {"default": 220}), # sensitivity
                "tempo": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}), # sensitivity
                "intensity": ("FLOAT", {"default": 0.75}),
                "smoothing": ("INT", {"default": 20}), # factor
                "hop_length": ("INT", {"default": 512}),

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
        positive, # What's a Conditioning?
        negative,
        seed: int,
        latent_image: torch.Tensor,
        seed_mode: str,
        latent_mode: str,
        pitch: int,
        tempo: float,
        intensity: float,
        smoothing: int,
        hop_length: int,

        model, # What's a Model?
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
    ):
        ## Setup Calculations
        # Unpack the audio
        y, sr = audio

        # Calculate parameters
        pitch = (300 - pitch) * 512 / hop_length
        tempo = tempo * hop_length / 512 # TODO: replace with librosa.feature.tempo

        if smoothing > 1:
            smoothing = int(smoothing * 512 / hop_length)

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

        # Calculate normalized power gradient per hop
        spectroGrad = self._normalizeArray(np.gradient(spectroMean))

        # Normalize the spectro mean
        spectroMean = self._normalizeArray(spectroMean)

        # Calculate pitch chroma for hops
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=hop_length
        )

        # Sort pitch chroma
        chromaSort = np.argsort(np.mean(chroma, axis=1))[::-1]

        # print("INPUT TENSOR:", latent_image["samples"], latent_image["samples"].shape)

        ## Generation
        # Prepare latent output tensor
        outputTensor: torch.Tensor = None
        for i in tqdm(range(len(spectroGrad)), desc="Music Visualization"):
            # TODO: Add option to iterate prompt

            # Calculate the latent tensor
            latentTensor = self._iterateLatentByMode(
                latent_image["samples"],
                latent_mode,
                intensity,
                tempo,
                spectroMean[i],
                spectroGrad[i],
                chromaSort
            )

            # print("LATENT TENSOR:", latentTensor, latentTensor.shape)
            print("LATENT MIN MAX:", torch.min(latentTensor), torch.max(latentTensor), torch.mean(latentTensor))

            # Generate the image
            imgTensor = common_ksampler(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    {"samples": latentTensor}, # ComfyUI, why package it?
                    denoise=denoise
                )[0]['samples']

            if outputTensor is None:
                outputTensor = imgTensor
            else:
                outputTensor = torch.vstack((
                    outputTensor,
                    # latentTensor,
                    imgTensor
                ))

            # Iterate seed as needed
            seed = self._iterateSeedByMode(seed, seed_mode)

            if i == 32: # TODO: remove (or add option for this?)
                break

        print(outputTensor)
        print(outputTensor.shape)
        for t in outputTensor:
            print(torch.min(t), torch.max(t), torch.mean(t))

        return ({"samples": outputTensor}, )

    # Private Functions
    def _normalizeArray(self, array: np.ndarray) -> np.ndarray:
        """
        Normalizes the given array.

        array: A numpy array.

        Returns a normalized numpy array.
        """
        minVal = np.min(array)
        return (array - minVal) / (np.max(array) - minVal)

    def _iterateLatentByMode(self,
        latent: torch.Tensor,
        latentMode: str,
        intensity: float,
        tempo: float,
        spectroMean: float,
        spectroGrad: float,
        chromaSort: float) -> torch.Tensor:
        """
        Produces a latent tensor based on the provided mode.

        latent: The latent tensor to modify.
        latentMode: The mode to iterate by.
        intensity: The amount to modify the latent by each hop.
        tempo: The tempo of the audio.
        spectroMean: The normalized mean power of the audio.
        spectroGrad: The normalized power gradient of the audio.
        chromaSort: The sorted pitch chroma of the audio.

        Returns the iterated latent tensor.
        """
        # Decide what to do if in flow mode
        if latentMode == MusicVisualizer.LATENT_MODE_FLOW:
            # Each hop will add or subtract, based on the audio features, from the last latent
            if random.random() < 0.5:
                latentMode = MusicVisualizer.LATENT_MODE_INCREASE
            else:
                latentMode = MusicVisualizer.LATENT_MODE_DECREASE

        # Decide what to do based on mode
        if latentMode == MusicVisualizer.LATENT_MODE_INCREASE:
            # Each hop adds, based on the audio features, to the last latent
            return self._applyFeatToLatent(latent, MusicVisualizer.FEAT_APPLY_METHOD_ADD, intensity, tempo, spectroMean, spectroGrad, chromaSort)
        elif latentMode == MusicVisualizer.LATENT_MODE_DECREASE:
            # Each hop subtracts, based on the audio features, from the last latent
            return self._applyFeatToLatent(latent, MusicVisualizer.FEAT_APPLY_METHOD_SUBTRACT, intensity, tempo, spectroMean, spectroGrad, chromaSort)
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

    def _applyFeatToLatent(self,
        latent: torch.Tensor,
        method: str,
        intensity: float,
        tempo: float,
        spectroMean: float,
        spectroGrad: float,
        chromaSort: float) -> torch.Tensor:
        """
        Applys the provided features to the latent tensor.

        latent: The latent tensor to modify.
        method: The method to use to apply the features.
        intensity: The amount to modify the latent by each hop.
        tempo: The tempo of the audio.
        spectroMean: The normalized mean power of the audio.
        spectroGrad: The normalized power gradient of the audio.
        chromaSort: The sorted pitch chroma of the audio.

        Returns the modified latent tensor.
        """
        # Apply features to every point in the latent
        if method == MusicVisualizer.FEAT_APPLY_METHOD_ADD:
            # Add the features to the latent
            latent += (tempo * (spectroMean + spectroGrad)) + intensity # Is this a good equation? Who knows!
        else: # FEAT_APPLY_METHOD_SUBTRACT
            # Subtract the features from the latent
            latent -= (tempo * (spectroMean + spectroGrad)) + intensity

        # TODO: wrap chromaSort over the whole thing

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
