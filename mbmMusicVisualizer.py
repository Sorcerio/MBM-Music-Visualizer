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
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_image": ("LATENT", ),

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

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("LATENTS", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/MusicVisualizer"

    def process(self,
        audio: tuple,
        positive, # What's a Conditioning?
        negative,
        seed: int,
        latent_image: torch.Tensor,

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

        for i in tqdm(range(len(spectroGrad)), desc="Music Visualization"):
            # Calculate the latent tensor
            latent_tensor = self._createLatent(tempo, spectroMean[i], spectroGrad[i], chromaSort).unsqueeze(0) # TODO: latent size from provided

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

            if i == 20:
                break

        # print(outputTensor)
        # print(outputTensor.shape)

        return ({"samples": outputTensor}, )

    def _normalizeArray(self, array: np.ndarray) -> np.ndarray:
            minVal = np.min(array)
            maxVal = np.max(array)
            return (array - minVal) / (maxVal - minVal)

    def _createLatent(self, tempo: float, spectroMean: float, spectroGrad: float, chromaSort: float):
        # Convert the inputs to numpy arrays and concatenate
        features = np.concatenate([
            np.array([tempo]),
            np.array([spectroMean]),
            np.array([spectroGrad]),
            # chromaSort
        ])

        # Ensure the features array is long enough
        if len(features) < 2*4*64*64:
            features = np.pad(features, (0, 2*4*64*64 - len(features)), "wrap")

        # Use the features to parameterize a multivariate normal distribution
        mean = torch.tensor(features[:4*64*64], dtype=torch.float32)
        stdDev = torch.tensor(features[4*64*64:2*4*64*64], dtype=torch.float32).abs() + 1e-7
        distribution = torch.distributions.Normal(mean, stdDev)

        # Sample from the distribution to create the latent tensor
        latentTensor = distribution.sample()

        # Reshape the latent tensor to the desired shape
        latentTensor = latentTensor.reshape((4, 64, 64))

        # Normalize the latent tensor to be between 0.0 and 1.0
        latentTensor = (latentTensor - torch.min(latentTensor)) / (torch.max(latentTensor) - torch.min(latentTensor))

        return latentTensor
