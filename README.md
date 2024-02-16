# MBM's Music Visualizer

![Header Image](./assets/header_v1.png "MBM Music Visualizer's Header Image")

>
> Nothing fuzzy about it.
>

- [MBM's Music Visualizer](#mbms-music-visualizer)
  - [What is it?](#what-is-it)
  - [Includes](#includes)
    - [Nodes](#nodes)
    - [Types](#types)
  - [Install](#install)
  - [Usage](#usage)

---

## What is it?

An image generation based music visualizer integrated into [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) as custom nodes.

## Includes

### Nodes

* `Audio Loader`: For loading an audio file using the [librosa](https://librosa.org) library.
* `Prompt Sequence Builder`: For stacking multiple prompts into a Prompt Sequence (`PROMPT_SEQ`).
* `Music Visualizer`: For rendering multiple images based on `AUDIO` input and other parameters.

### Types

* `AUDIO`: For representing loaded audio files.
* `PROMPT_SEQ`: For representing multiple prompts (positive and negative) in an ordered sequence.

## Install

1. Enter ComfyUI's Python Environment by running `.\.venv\Scripts\activate` from ComfyUI's root directory.
2. Clone this repo into ComfyUI's `custom_nodes` directory by entering the directory and running: `git clone <LINK_TBD> MBM_MusicVisualizer`.
3. Enter the `MBM_MusicVisualizer` directory.
4. Run `pip install -r .\requirements.txt` to install this project's dependencies.
5. Start ComfyUI as normal.

Nodes will be found in the `MBMnodes/` submenu inside ComfyUI.

## Usage

![Example Music Visualizer Flow](./assets/ExampleMusicVisualizer.png "Example Music Visualizer Flow")
> 📝 **Note:** Drag the above image (or [the example workflow file](./assets/ExampleMusicVisualizer.json)) into ComfyUI to automatically load this flow!

Place any audio files you would like to load in the [audio/](./audio/) directory.
You can always refresh the webpage ComfyUI is loaded into to refresh the list in the `Audio Loader` node.

The `Music Visualizer` node takes an `AUDIO` object in and produces a set of Latent Images on output.
A `tqdm` progress bar will be shown in the console to display the current status of the visualization and how long it is expected to take.

Upon completion of a visualization, the `Music Visualizer` will output the input FPS, a set of Latent Images that can be decoded using the built-in Latent Decoder, and a set of Images showing relevant data from the run.
The FPS can be fed into any further video or gif generating nodes.
The Latent Images, which are the output content of the visualization, should be converted and either saved individually or compiled into a video through another node.
The charts are pixel images that can be saved or modified as desired.

_When testing your generations_, consider setting the `image_limit` to `1` or higher to generate only a specific number of images.
Doing so **will still produce complete charts** for most data sources allowing you to preview the general flow of the visualization before you commit to image generation for all frames.
