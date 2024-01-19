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
                "tempo": ("INT", {"default": 220}), # sensitivity
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
        tempo: int,
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
            hop_length=hop_length
        )

        # Sort pitch chroma
        chromaSort = np.argsort(np.mean(chroma, axis=1))[::-1]

        ## Generation
        # Prepare latent output list
        outputLatents: list[torch.Tensor] = []

        # Calculate tensor length
        tensorLen = len(latent_image)

        # Report
        print(f"Generating {tensorLen} images for music visualization")

        # Loop through the spectrogram gradient
        curLatent = latent_image['samples']
        curJitter: np.ndarray = self._addJitter(jitter, tensorLen)
        latentUpdateLast = np.zeros(tensorLen)
        for i in tqdm(range(len(spectroGrad)), desc="Generating"):
            ## Latent Noise modification
            # Calculate some jitters
            if ((i % 200) == 0):
                curJitter = self._addJitter(jitter, tensorLen)

            # Calculate the latent update
            latentUpdate = np.array([tempo for k in range(tensorLen)]) * (spectroGrad[i] + spectroMean[i]) * curJitter # TODO: didn't do 'update_dir'

            # Smooth the latent update relative to last
            latentUpdate = (latentUpdate + latentUpdateLast * 3) / 2

            # Record the latent
            curLatent = curLatent + latentUpdate
            latentUpdateLast = latentUpdate

            ## Prompt modification
            # TODO: add this

            ## Generation
            # Generate the image
            outputLatents.append(
                common_ksampler(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    {"samples": curLatent}, # ComfyUI, why package it?
                    denoise=denoise
                )[0]
            )

        return (outputLatents, )

    def _addJitter(self, jitterMod: float, length: int) -> np.ndarray:
        """
        Adds jitter to the given array's values.

        jitterMod: The amount of jitter modification to apply.
        length: The desired length of the output array.

        Returns the jittered array.
        """
        # Add jitter
        jitterOut = np.zeros(length)
        for i in range(length):
            if random.uniform(0, 1) < 0.5:
                jitterOut[i] = 1
            else:
                jitterOut[i] = 1 - jitterMod

        return jitterOut
