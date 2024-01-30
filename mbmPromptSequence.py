# MBM's Music Visualizer: Prompt Sequence
# Allows for building of a sequence of prompts.

# Imports
import torch

# Classes
class PromptSequenceBuilder:
    """
    Allows for building of a sequence of prompts.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", )
            },
            "optional": {
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "prompts": ("PROMPT_SEQ", )
            }
        }

    RETURN_TYPES = ("PROMPT_SEQ", )
    RETURN_NAMES = ("PROMPTS", )
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Prompts"

    def process(self, positive_1, negative_1, positive_2 = None, negative_2 = None, prompts: torch.Tensor = None):
        """
        Returns a Tensor of the shape `[num of prompt sets, 2, *input tensor shape]` where 2 indicates the positive and negative tensors.
        """
        # Create a new prompt sequence
        if (positive_2 is None) or (negative_2 is None):
            # Include just the required one
            promptsOut = torch.stack((positive_1[0][0].squeeze(), negative_1[0][0].squeeze())).unsqueeze(0)
        else:
            # Include both
            promptsOut = torch.stack((
                torch.stack((positive_1[0][0].squeeze(), negative_1[0][0].squeeze())),
                torch.stack((positive_2[0][0].squeeze(), negative_2[0][0].squeeze()))
            ))

        # Add to the given prompt sequence if provided
        if prompts is not None:
            promptsOut = torch.cat((prompts, promptsOut), dim=0)

        return (promptsOut, )
