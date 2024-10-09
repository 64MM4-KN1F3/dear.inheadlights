---
title: Infinite Cities by Ita-Llama Calvino - Part 1
publishDate: 2024-10-06
tags: ["LLMs", "AI", "LLM fine-tuning", "italo calvino", "invisible cities"]
---

![Italo Calvino - Invisible Cities](../../assets/invisible-cities-cover-cropped.png)

**TLDR; Fine-tune an LLM to imitate a dead author! Examples** <a class="text-slate-700 dark:text-slate-300 hover:text-rainbow font-semibold" href='/tag/infinite cities'>here</a>.

We'll approach this in stages. Skip ahead if you like! This post is primarily environment setup and data preparation. Jump to [part 2](../italo-calvino-infinite-cities-pt2) for fine-tuning.


## Preamble

Six months ago I bought a copy of Italo Calvino's [Invisible Cities](https://www.amazon.com.au/Invisible-Cities-Italo-Calvino/dp/0099429837?crid=13Y6OV6MGL0L9&dib=eyJ2IjoiMSJ9.mEsvjBLCE7EN69Worao3kVl6gVWLPdrEku2cT7yHiYoaPVopCBLGqufzn_ej4tCGk9MchJxT0hHfg1WrCfmQ0vXJeczt_ol81_4RHhL76FYu5Kq3irVAVLr6ADpF-onlujRoM9qecxOkJLT6fZZTMd6N0VARYqw6YOtmh-oWZPLQs4LFSy0Bgbh3UIQqQqvRMpOSK9ab3sX67CV62_kW-3muMvDqYLOI9dZv2T1eTQyLL-jkI0hc5XqYJzPm-HfCbBnpO3BkFO008Ei3MrBfzXfxqEeDqNpebSJlc0fbQDA._ljdGS5TgElq9-l7lJKiclR4CQlUQQdVpJpvsJ7lKp8&dib_tag=se&keywords=italo+calvino&qid=1728280495&sprefix=italo+calvino%2Caps%2C353&sr=8-2&linkCode=ll1&tag=dearinheadlig-22&linkId=ad2e500caf7d4a073a1fc9d4a59e1e71&language=en_AU&ref_=as_li_ss_tl), it wasn't long before I fell for the whimsical melancholy inside the pages. I managed to lose the book a week or so later but not before visiting each of the 56 cities as relayed by Marco Polo to the aging Kublai Khan.

56 weird and wonderful city descriptions is quite a lot, but I found myself wanting more, which brings me here, into LLM-land on a quest to generate *more* cities. I want Infinite Cities! Perhaps Kublai Khan would have too.

While the current state-of-the-art LLMs tend to be closed-source, 'open-ish' models are an exciting frontier with new releases dropping week to week. This week NVIDIA released a new [multimodal model](https://huggingface.co/nvidia/NVLM-D-72B) for example.

## Ingredients

To fine-tune an LLM you need the following broad ingredients:
- A base model to train on top of
- Reasonably generous computing oomph
- Data to train with

In my case:
- Mistral-7B-v0.3 - https://huggingface.co/mistralai/Mistral-7B-v0.3
- Macbook Pro M2
- Digitised text of Invisible Cities as CSV

I may not cover them all, but the software ingredients we'll make use of are:
- [VSCode](https://code.visualstudio.com/download)
- [homebrew](https://docs.brew.sh/Installation)
- [git](https://git-scm.com/downloads)
- python - 🐍
- pytorch - 🔥
- pandas - 🐼
- [uv](https://github.com/astral-sh/uv) - a fancy new package manager for python
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ LLM inference library (and other useful tools)
- [ollama](https://ollama.com) - LLM CLI and server
- [OpenWebUI](https://docs.openwebui.com/getting-started/) - a great front-end for ollama
- [Docker](https://www.docker.com/products/docker-desktop/) - used as a virtual environment for OpenWebUI

## Post-preamble

Six months in the future is a long time these days. Keep that in mind if you read this guide. Things tend to evolve quickly in the LLM space. What worked today (2024-10-06) may not work the same way tomorrow.

This post will be MacOS-focussed but should extrapolate to other linux-flavoured environments. I haven't tinkered with LLMs in Windows environments but perhaps using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about) is one way to follow along in the Microsoft world. Your mileage may vary.

## LLM Environment

Running local LLMs on Mac Silicon with llama.cpp, Ollama and OpenWebUI is a great experience. Grab Ollama from [here](https://ollama.com/download) and it will install llama.cpp under the covers for you however I do recommend cloning a separate copy of llama.cpp as we'll be using specific tools from that repo to convert our fine-tuned models from .safetensors to .gguf later in this tutorial.

Clone llama.cpp (put it somewhere you'll remember by substituting `/path/to/` with an appropriate location):
```shell
% git clone https://github.com/ggerganov/llama.cpp.git /path/to/llama.cpp
```
While we're at it, let's get a copy of the latest Mistral7B model that we'll use as our base for fine-tuning. Again, chose an appropriate path, we'll need this later:
```shell
% git clone https://huggingface.co/mistralai/Mistral-7B-v0.3 /path/to/Mistral-7B-v0.3
```
I highly recommend Open WebUI as a web front end to Ollama. The OpenWebUI team recommend running it in a docker container, so [download](https://www.docker.com/products/docker-desktop/) and install Docker Desktop, then use the following script to install OpenWebUI. You can also use the same script to update OpenWebUI later!
```shell
docker pull ghcr.io/open-webui/open-webui:main
docker stop open-webui
docker rm open-webui
docker run -d -p 3000:8080 -e WEBUI_AUTH=false --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```
Bundle the above into a shell script called `update-OpenWebUI.sh` if you like to use later. Remember to make that script excecutable with:
```shell
% chmod +x update-OpenWebUI.sh
```

Now direct your web browser to `http://localhost:3000` to access OpenWebUI. 

## Tuning Environment

Python is great but I can't remember the number of times I've ended up in python-dependency hell. To avoid this I recommend the use of uv to set up your environment. uv rolls together package management, environment management and other much-needed quality-of-life capabilities for python scripting and projects.

Grab [uv](https://github.com/astral-sh/uv) with homebrew (or OS package manager of your choice):
```shell
% brew install uv
```
Create your project home then we'll jump into VSCode:
```shell
% mkdir your_project
% cd your_project
% code .
```
In VSCode, open a terminal with <kbd>CTRL</kbd> <kbd>SHIFT</kbd> <kbd>\`</kbd> or via the menu: *Terminal > New Terminal*

In the terminal window initialise uv:
```shell
% uv init
```
Let's install a version of python supported by pytorch:
```shell
% uv python install '>=3.8'
```
Create the uv environment then activate it:
```shell
% uv venv
% source .venv/bin/activate
```
*(later on you can exit the virtual environment with the `deactivate` command)*

Now we can download and add the python packages we'll use:
```shell
% uv add pandas tqdm torch transformers
```
That step may take a moment so stand up if you're sitting down, or sit down if you're standing up.

Now is a good time to initialise your git environment (`brew install git` if you don't aready have git installed):
```shell
% git init
```
Have a look in the .gitignore file that uv has created. This prevents certain files and folders from being tracked by git. Super important for anything credential-related, not that we're using any services that need us to logon in this project, one of the benefits of playing with local LLMs!

Make your first commit:
```shell
% git add *
% git commit -m "First commit for fine-tuning scripts"
```

<sup>*</sup>*If you've configured a remote repository you may as well push that first commit now:*
```shell
% git push
```
Make git commits as you follow along, from here on I'll assume you're doing so.

## Project structure
As a reference, this is what your environment will look like eventually after we've added all the necessary scripts:
```shell
your_project/
├── .venv/ # Ignore, uv will manage this
├── .git/ # Ignore, git will manage this
├── .gitignore # Add anything you don't want to track with git
├── .python-version # Ignore, uv will manage this
├── dataset_preparation.py # TBC
├── fine_tune_model.py # TBC
├── invisible_cities.csv # Raw training data
├── training_data.csv # Formatted training data with 'input' and 'output' columns
├── train_dataset/ # Generated by dataset_preparation.py
├── eval_dataset/ # Generated by dataset_preparation.py
├── fine-tune-01/ # Generated by peft_fine_tune_model.py
├── .pyproject.toml # Mostly ignore. This is uv's list of project requirements
└── uv.lock # Ignore, uv will manage this
```

## Source training data

This you'll need to find yourself and pop into a CSV with a couple of column titles, 'input' and 'output'. In my case (this is abbreviated to one city):

| input                            | output                                                        |
|----------------------------------|---------------------------------------------------------------|
| "Cities and memory - Zora"       | "Beyond six rivers and three mountain ranges rises Zora...    |
| "Cities and signs - Zirma"       | "Travelers return from the city of Zirma with..."             |

With fine-tuning, the more records you have in your training data the better. I may make a later post about synthesising additional training data at some stage.

## Preparing the Dataset

Now that our environment is set up and we're comfortably seated (or standing, if you took my earlier advice), it's time to prepare our dataset. Let's create a python script to do this:

```python
# dataset_preparation.py

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# Load the processed data
data = pd.read_csv('processed_data.csv')

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(data[['input', 'output']])

# Split the dataset into training and validation sets
dataset = dataset.train_test_split(test_size=0.1)

# Save the datasets for later use
dataset['train'].save_to_disk('train_dataset')
dataset['test'].save_to_disk('eval_dataset')
```

Let's walk through what this script does:

1. **Import Libraries**: We start by importing the necessary libraries.
   - `pandas` for handling our CSV data.
   - `Dataset` from Hugging Face's [`datasets`](https://pypi.org/project/datasets/) library to manage our dataset efficiently.
   - `tqdm` for progress bars, which is handy if you're dealing with large amounts of data.

2. **Load the Data**: We read the `processed_data.csv` file into a pandas DataFrame.
   ```python
   data = pd.read_csv('processed_data.csv')
   ```
   Make sure your CSV file is in the same directory as your script or provide the correct path.

3. **Create a Hugging Face Dataset**: We convert the DataFrame into a Hugging Face `Dataset` object, focusing only on the `input` and `output` columns.
   ```python
   dataset = Dataset.from_pandas(data[['input', 'output']])
   ```
   This format is optimal for training with Hugging Face's tools.

4. **Split the Dataset**: We split the dataset into training and validation sets using an 90/10 split.
   ```python
   dataset = dataset.train_test_split(test_size=0.1)
   ```
   This helps us evaluate the model's performance on unseen data.

5. **Save the Datasets**: We save both the training and validation datasets to disk for later use.
   ```python
   dataset['train'].save_to_disk('train_dataset')
   dataset['test'].save_to_disk('eval_dataset')
   ```
   This creates two directories, `train_dataset` and `eval_dataset`, containing our processed data.

To run the script, make sure your virtual environment is activated and execute:

```shell
% uv run dataset_preparation.py
```
With our dataset prepared, we're one step closer to generating infinite cities. 

To be continued in [part 2](../italo-calvino-infinite-cities-pt2)!
# 🤖