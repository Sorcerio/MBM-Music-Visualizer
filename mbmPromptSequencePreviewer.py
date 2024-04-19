# MBM's Music Visualizer: Prompt Sequence Previewer
# A utility node to render each prompt in a prompt sequence as a standalone image.

# Imports
import torch
from tqdm import tqdm

import comfy.samplers
from nodes import common_ksampler

from .mbmPrompt import MbmPrompt

# Classes
class PromptSequencePreviewer:
    """
    A utility node to render each prompt in a prompt sequence as a standalone image.

    Commonly used for previewing the output of a prompt sequence's prompts individually.
    """
    # Class Constants
    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("LATENTS", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Prompts"

    # ComfyUI Functions
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Visualizer Settings
                "prompts": ("PROMPT_SEQ", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_image": ("LATENT", ),
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
            seed: int,
            latent_image: dict[str, torch.Tensor],
            model,
            steps: int,
            cfg: float,
            sampler_name: str,
            scheduler: str,
            denoise: float,
        ):
        # Loop through the prompts
        outputTensor: torch.Tensor = None
        for prompt in (pbar := tqdm(prompts, desc="Rendering Prompts")):
            # Generate the image
            imgTensor = common_ksampler(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                prompt.positivePrompt(),
                prompt.negativePrompt(),
                latent_image,
                denoise=denoise
            )[0]["samples"]

            if outputTensor is None:
                outputTensor = imgTensor
            else:
                outputTensor = torch.vstack((
                    outputTensor,
                    imgTensor
                ))

        # Return outputs
        return ({"samples": outputTensor}, )
