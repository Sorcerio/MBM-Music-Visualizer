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
    SEED_MODE_FIXED = "fixed" # Seed stays the same
    SEED_MODE_RANDOM = "random" # Seed is random every hop
    SEED_MODE_INCREASE = "increase" # Seed increases by 1 every hop
    SEED_MODE_DECREASE = "decrease" # Seed decreases by 1 every hop

    LATENT_MODE_STATIC = "static" # Only the provided latent is used ignoring audio features
    LATENT_MODE_INCREASE = "increase" # Each hop adds, based on the audio features, to the last latent
    LATENT_MODE_DECREASE = "decrease" # Each hop subtracts, based on the audio features, from the last latent
    LATENT_MODE_FLOW = "flow" # Each hop will add or subtract, based on the audio features, from the last latent
    LATENT_MODE_EACHHOP = "each_hop" # Each hop creates a new latent based on the audio features

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_image": ("LATENT", ),

                "seed_mode": ([MusicVisualizer.SEED_MODE_FIXED, MusicVisualizer.SEED_MODE_RANDOM, MusicVisualizer.SEED_MODE_INCREASE, MusicVisualizer.SEED_MODE_DECREASE], ),
                "latent_mode": ([MusicVisualizer.LATENT_MODE_FLOW, MusicVisualizer.LATENT_MODE_STATIC, MusicVisualizer.LATENT_MODE_INCREASE, MusicVisualizer.LATENT_MODE_DECREASE, MusicVisualizer.LATENT_MODE_EACHHOP], ),
                # TODO: Add latent jitter (amount to modify the latent each hop by)

                # TODO: Move these into a KSamplerSettings node
                # Also might be worth adding a KSamplerSettings to KSamplerInputs node that splits it all out to go into the standard KSampler when done here?
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # TODO: Move these into a MusicVisualizerSettings node
                "pitch": ("INT", {"default": 220}), # sensitivity
                "tempo": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}), # sensitivity
                "depth": ("FLOAT", {"default": 1.0}),
                "jitter": ("FLOAT", {"default": 0.5}),
                "truncation": ("FLOAT", {"default": 1.0}),
                "smoothing": ("INT", {"default": 20}), # factor
                "hop_length": ("INT", {"default": 512}),
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

        model, # What's a Model?
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,

        pitch: int,
        tempo: float,
        depth: float,
        jitter: float,
        truncation: float,
        smoothing: int,
        hop_length: int
    ):
        ## Setup Calculations
        # Unpack the audio
        y, sr = audio

        # Calculate parameters
        pitch = (300 - pitch) * 512 / hop_length
        tempo = tempo * hop_length / 512

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

        ## Generation
        # Prepare latent output tensor
        outputTensor: torch.Tensor = None
        latentSize = latent_image["samples"].shape
        for i in tqdm(range(len(spectroGrad)), desc="Music Visualization"):
            # TODO: Add option to iterate prompt

            # Calculate the latent tensor
            # TODO: latent size from provided
            # TODO: iterate on top of previous latent!
            # TODO: make optional
            latent_tensor = self._createLatentWithFeats(latentSize, tempo, spectroMean[i], spectroGrad[i], chromaSort).unsqueeze(0)

            print("LATENT TENSOR:", latent_tensor, latent_tensor.shape)
            print("LATENT MIN MAX:", torch.min(latent_tensor), torch.max(latent_tensor))

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
                    {"samples": latent_tensor}, # ComfyUI, why package it?
                    denoise=denoise
                )[0]['samples']

            if outputTensor is None:
                outputTensor = imgTensor
            else:
                outputTensor = torch.vstack((
                    outputTensor,
                    # latent_tensor,
                    imgTensor
                ))

            # Iterate seed as needed
            seed = self._iterateSeedByMode(seed, seed_mode)

            if i == 20: # TODO: remove (or add option for this?)
                break

        # print(outputTensor)
        # print(outputTensor.shape)

        return ({"samples": outputTensor}, )

    # Private Functions
    def _normalizeArray(self, array: np.ndarray) -> np.ndarray:
            """
            Normalizes the given array.

            array: A numpy array.

            Returns a normalized numpy array.
            """
            minVal = np.min(array)
            maxVal = np.max(array)
            return (array - minVal) / (maxVal - minVal)

    def _createLatentWithFeats(self, shape: tuple[int, int, int, int], tempo: float, spectroMean: float, spectroGrad: float, chromaSort: float) -> torch.Tensor:
        """
        Creates a latent tensor from the provided audio features.

        shape: The shape of the latent tensor.
        tempo: The tempo of the audio.
        spectroMean: The normalized mean power of the audio.
        spectroGrad: The normalized power gradient of the audio.
        chromaSort: The sorted pitch chroma of the audio.

        Returns a latent tensor (without the "batch" dimension; add this with `.unsqueeze(0)`).
        """
        # Convert the inputs to numpy arrays and concatenate
        features = np.concatenate([
            np.array([tempo]),
            np.array([spectroMean]),
            np.array([spectroGrad]),
            # chromaSort
        ])

        print(tempo, spectroMean, spectroGrad)

        print("FEATURES 1:", features, features.shape)

        # Calculate the array bounds
        boundsMain = shape[1] * shape[2] * shape[3]
        boundsDouble = shape[0] * boundsMain

        # Ensure the features array is long enough
        if len(features) < boundsDouble:
            features = np.pad(features, (0, boundsDouble - len(features)), "wrap")

        print("FEATURES 2:", features, features.shape)

        # Use the features to parameterize a multivariate normal distribution
        mean = torch.tensor(features[:boundsMain], dtype=torch.float32)
        stdDev = torch.tensor(features[boundsMain:boundsDouble], dtype=torch.float32).abs() + 1e-7
        distribution = torch.distributions.Normal(mean, stdDev)

        # Sample from the distribution to create the latent tensor
        latentTensor = distribution.sample()

        print("FEATURES 3:", features, features.shape)

        # Reshape the latent tensor to the desired shape
        latentTensor = latentTensor.reshape((shape[1], shape[2], shape[3]))

        # Normalize the latent tensor to be between 0.0 and 1.0
        latentTensor = (latentTensor - torch.min(latentTensor)) / (torch.max(latentTensor) - torch.min(latentTensor))

        print("FEATURES 4:", features, features.shape)

        exit()

        return latentTensor

    def _iterateSeedByMode(self, seed: int, seedMode: str):
        """
        Creates a new seed based on the provided mode.

        seed: The seed to iterate.
        seedMode: The mode to iterate by.

        Returns the iterated seed.
        """
        if seedMode == MusicVisualizer.SEED_MODE_FIXED:
            return seed
        elif seedMode == MusicVisualizer.SEED_MODE_RANDOM:
            return random.randint(0, 0xffffffffffffffff)
        elif seedMode == MusicVisualizer.SEED_MODE_INCREASE:
            return seed + 1
        elif seedMode == MusicVisualizer.SEED_MODE_DECREASE:
            return seed - 1
        else:
            return seed
