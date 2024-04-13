# MBM's Music Visualizer: Prompt Sequence Renderer
# Renders a prompt sequence into a set of images.

# Imports
import torch
import random
import numpy as np
from tqdm import tqdm

import comfy.samplers
from nodes import common_ksampler

from .mbmPrompt import MbmPrompt
from .mbmMVShared import chartData

# Classes
class PromptSequenceRenderer:
    """
    Renders a prompt sequence into a set of images.
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

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("LATENTS", "CHARTS")
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Prompts"

    # Constructor
    def __init__(self):
        self.__isBouncingUp = True # Used when `bounce` mode is used to track direction

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Visualizer Settings
                "prompts": ("PROMPT_SEQ", ),
                "latent_mods": ("TENSOR_1D", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_image": ("LATENT", ),
                "seed_mode": ([s.SEED_MODE_FIXED, s.SEED_MODE_RANDOM, s.SEED_MODE_INCREASE, s.SEED_MODE_DECREASE], ),
                "latent_mode": ([s.LATENT_MODE_BOUNCE, s.LATENT_MODE_FLOW, s.LATENT_MODE_STATIC, s.LATENT_MODE_INCREASE, s.LATENT_MODE_DECREASE, s.LATENT_MODE_GAUSS], ),
                "image_limit": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}), # Provide `<= 0` to use whatever audio sampling comes up with
                "latent_mod_limit": ("FLOAT", {"default": 5.0, "min": -1.0, "max": 10000.0}), # The maximum variation that can occur to the latent based on the latent's mean value. Provide `<= 0` to have no limit

                # Generation Settings
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self,
            prompts: list[MbmPrompt],
            latent_mods: torch.Tensor,
            seed: int,
            latent_image: dict[str, torch.Tensor],
            seed_mode: str,
            latent_mode: str,
            image_limit: int,
            latent_mod_limit: float,

            model,
            steps: int,
            cfg: float,
            sampler_name: str,
            scheduler: str,
            denoise: float,
        ):
        ## Setup
        # Set the random library seeds
        random.seed(seed)
        np.random.default_rng(seed)

        # Get the counts
        desiredFrames = len(prompts)

        ## Validation
        # Make sure if bounce mode is used that the latent mod limit is set
        if (latent_mode == self.LATENT_MODE_BOUNCE) and (latent_mod_limit <= 0):
            raise ValueError("Latent Mod Limit must be set to `>0` when using the `bounce` Latent Mode")

        # Check if prompts are provided
        if desiredFrames < 1:
            raise ValueError("At least one prompt is required.")

        ## Generation
        # Reset the bounce direction
        self.__isBouncingUp = True

        # Set the initial prompt
        promptPos = prompts[0].positivePrompt()
        promptNeg = prompts[0].negativePrompt()

        # Prepare latent output tensor
        outputTensor: torch.Tensor = None
        latentTensorMeans = np.zeros(desiredFrames)
        latentTensor = latent_image["samples"].clone()
        for i in (pbar := tqdm(range(desiredFrames), desc="Rendering Sequence")):
            # Calculate the latent tensor
            latentTensor = self._iterateLatentByMode(
                latentTensor,
                latent_mode,
                latent_mod_limit,
                latent_mods[i]
            )

            # Records the latent tensor's mean
            latentTensorMeans[i] = torch.mean(latentTensor).numpy()

            # Set progress bar info
            pbar.set_postfix({
                "mod": f"{latent_mods[i]:.2f}",
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
                    imgTensor
                ))

            # Limit if one if supplied
            if (image_limit > 0) and (i >= (image_limit - 1)):
                break

            # Iterate seed as needed
            seed = self._iterateSeedByMode(seed, seed_mode)

            # Iterate the prompts as needed
            if (desiredFrames > 1) and ((i + 1) < desiredFrames):
                promptPos = prompts[i + 1].positivePrompt()
                promptNeg = prompts[i + 1].negativePrompt()

        # Render charts
        chartImages = torch.vstack([
            chartData(latentTensorMeans, "Latent Means")
        ])

        # Return outputs
        return (
            {"samples": outputTensor},
            chartImages
        )

    # Internal Functions
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
        if latentMode == self.LATENT_MODE_FLOW:
            # Each hop will add or subtract, based on the audio features, from the last latent
            if random.random() < 0.5:
                latentMode = self.LATENT_MODE_INCREASE
            else:
                latentMode = self.LATENT_MODE_DECREASE

        if latentMode == self.LATENT_MODE_BOUNCE:
            # Increases to to the `modLimit`, then decreases to `-modLimit`, and loops as many times as needed building on the last latent
            if modLimit > 0:
                # Do the bounce operation
                # Calculate the next value
                curLatentMean = torch.mean(latent)
                nextValue = (curLatentMean + modifier) if self.__isBouncingUp else (curLatentMean - modifier)

                # Check if within bounds
                if -modLimit <= nextValue <= modLimit:
                    # Within bounds
                    latentMode = self.LATENT_MODE_INCREASE if self.__isBouncingUp else self.LATENT_MODE_DECREASE
                else:
                    # Outside of bounds
                    latentMode = self.LATENT_MODE_DECREASE if self.__isBouncingUp else self.LATENT_MODE_INCREASE
                    self.__isBouncingUp = not self.__isBouncingUp
            else:
                # No limit so just increase
                latentMode = self.LATENT_MODE_INCREASE

        # Decide what to do based on mode
        if latentMode == self.LATENT_MODE_INCREASE:
            # Each hop adds, based on the audio features, to the last latent
            return self._applyFeatToLatent(latent, self.FEAT_APPLY_METHOD_ADD, modLimit, modifier)
        elif latentMode == self.LATENT_MODE_DECREASE:
            # Each hop subtracts, based on the audio features, from the last latent
            return self._applyFeatToLatent(latent, self.FEAT_APPLY_METHOD_SUBTRACT, modLimit, modifier)
        elif latentMode == self.LATENT_MODE_GAUSS:
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
        # TODO: More specific noise range input?
        return torch.tensor(np.random.normal(3, 2.5, size=size))

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
        if method == self.FEAT_APPLY_METHOD_ADD:
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
        if seedMode == self.SEED_MODE_RANDOM:
            # Seed is random every hop
            return random.randint(0, 0xffffffffffffffff)
        elif seedMode == self.SEED_MODE_INCREASE:
            # Seed increases by 1 every hop
            return seed + 1
        elif seedMode == self.SEED_MODE_DECREASE:
            # Seed decreases by 1 every hop
            return seed - 1
        else: # SEED_MODE_FIXED
            # Seed stays the same
            return seed
