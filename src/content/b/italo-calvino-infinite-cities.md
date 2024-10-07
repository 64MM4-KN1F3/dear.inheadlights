---
title: Infinite Cities by Itallama Calvino
publishDate: 2024-10-06
tags: ["LLMs", "AI", "LLM fine-tuning", "italo calvino", "invisible cities"]
imagePath: ../../assets/invisible-cities-cover-cropped.png
altText: Italo Calvino - Invisible Cities
---

![Italo Calvino - Invisible Cities](../../assets/invisible-cities-cover-cropped.png)

**TLDR; Fine-tune an LLM to imitate a dead author! Examples** <a class="text-slate-700 dark:text-slate-300 hover:text-rainbow font-semibold" href='/tag/infinite cities'>here</a>.

We'll approach this in stages. Skip ahead if you like!


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

## Environment

Python is great but I can't remember the number of times I've ended up in python-dependency hell. To avoid this I recommend the use of uv to set up your environment. uv rolls together package management, package management and other much-needed quality-of-life capabilities for python scripting and projects.

Grab uv with homebrew (or OS package manager of your choice):
```shell
% brew install uv
```
Create your project home then we'll jump into VSCode:
```shell
% mkdir your_project
% code .
```
In VSCode, open a terminal with <kbd>CTRL</kbd> + <kbd>SHIFT</kbd> + <kbd>\`</kbd> or via the menu: *Terminal > New Terminal*

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

As a reference, this is what your environment will look like eventually after we've added all the necessary scripts:
```shell
your_project/
├── .venv/ # Ignore, uv will manage this
├── .gitignore # Mostly ignore, uv will manage this. But add anything you don't want to track.
├── .python-version # Ignore, uv will manage this
├── dataset_preparation.py # TBC
├── generate_descriptions.py # TBC
├── fine_tune_model_peft.py # TBC
├── invisible_cities.csv # Your training data
├── processed_data.csv # Generated by data_preparation.py
├── train_dataset/ # Generated by dataset_preparation.py
├── eval_dataset/ # Generated by dataset_preparation.py
├── fine-tune-01/ # Generated by peft_fine_tune_model.py
├── .pyproject.toml # Mostly ignore. This is uv's list of project requirements
└── uv.lock # Ignore, uv will manage this
```

## Source training data

This you'll need to find yourself and pop into a CSV with a couple of column titles, 'input' and 'output'. In my case (this is abbreviated to one city):
```csv
input,output
Cities and memory - Diomira, "Leaving there and proceeding for three days toward the east, you reach Diomira, a city with sixty silver domes, bronze statues of all the gods, streets paved with lead, a crystal theater, a golden cock that crows each morning on a tower. All these beauties will already be familiar to the visitor, who has seen them also in other cities. But the special quality of this city for the man who arrives there on a September evening, when the days are growing shorter and the multicolored lamps are lighted all at once at the doors of the food stalls and from a terrace a woman's voice cries ooh!, is that he feels envy toward those who now believe they have once before lived an evening identical to this and who think they were happy, that time."
```

## Preparing the Dataset

Now that our environment is set up and we're comfortably seated (or standing, if you took my earlier advice), it's time to dive into preparing our dataset. Remember that digitised text of *Invisible Cities* I mentioned? We'll transform it into a format suitable for fine-tuning our LLM.

First things first, ensure you have a `processed_data.csv` file in your project directory. This CSV should contain two columns: `input` and `output`. Think of `input` as the prompt or question you might pose to the model, and `output` as the response you'd expect—much like Marco Polo's descriptions to Kublai Khan.

Let's create a Python script to prepare our dataset:

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
   - `Dataset` from Hugging Face's `datasets` library to manage our dataset efficiently.
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

Preparing your data is like laying the foundation for a city. A shaky foundation leads to a shaky city, and we wouldn't want our infinite cities to crumble, would we? Ensure your `processed_data.csv` is clean and formatted correctly. Each row should represent a unique interaction:

- **Input**: The prompt or question for the model.
- **Output**: The expected response.

For example:

| input                            | output                                                        |
|----------------------------------|---------------------------------------------------------------|
| "Describe the city of Zaira."    | "Zaira is a city that..."                                     |
| "Tell me about the bridges of..."| "In the city of Octavia, bridges are..."                      |

With our dataset prepared, we're one step closer to generating infinite cities.

## Fine-Tuning the Model

With our dataset prepared, it's time to embark on fine-tuning the language model to generate our own "infinite cities." We'll be using a Python script called `fine_tune_model.py`, which leverages parameter-efficient fine-tuning (PEFT) with Low-Rank Adaptation (LoRA). This approach allows us to fine-tune large models on hardware with limited resources—like our MacBook Pro M2.

Here's the script:

```python
# fine_tune_model.py

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# Training iteration
iteration = "01"  # Increased tokenizer max length to 512

# Set the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load the datasets
train_dataset = load_from_disk('train_dataset')
eval_dataset = load_from_disk('eval_dataset')

# Use a smaller model
model_name = '/Users/alexweatherley/Documents/LLMs/Mistral-7B-v0.3'  # Adjust the path to your model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model.enable_input_require_grads()

# Apply PEFT with LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Verify model parameters require gradients
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Tokenization function
def tokenize_function(examples):
    inputs = []
    for i in tqdm(range(len(examples['input'])), desc='Tokenizing'):
        prompt = examples['input'][i]
        response = examples['output'][i]
        full_text = prompt + '\n' + response
        inputs.append(full_text)
    
    tokenized_outputs = tokenizer(
        inputs,
        truncation=True,
        max_length=512,  # Increased max_length to 512
        padding='max_length',
    )
    tokenized_outputs['labels'] = tokenized_outputs['input_ids'].copy()
    return tokenized_outputs

# Tokenize the datasets
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['input', 'output'],
    desc='Tokenizing Training Dataset'
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['input', 'output'],
    desc='Tokenizing Evaluation Dataset'
)

# Initialize data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results-' + iteration,
    eval_strategy='steps',
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Adjusted batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.0,
    save_total_limit=2,
    fp16=False,  # Disable fp16 mixed precision
    bf16=True,   # Enable bf16 mixed precision
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model('./fine-tune-' + iteration)
tokenizer.save_pretrained('./fine-tune-' + iteration)
```

Let's walk through the script step by step.

**Importing Libraries**

```python
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
```

- **torch**: For tensor operations and handling the device (CPU or GPU).
- **datasets**: To load our preprocessed datasets from disk.
- **transformers**: Provides the model, tokenizer, and training utilities from Hugging Face.
- **peft**: Implements parameter-efficient fine-tuning methods like LoRA.
- **tqdm**: For progress bars during tokenization.

**Setting Up the Training Iteration and Device**

```python
iteration = "01"  # Increased tokenizer max length to 512

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

- **iteration**: A variable to keep track of training runs, useful if you plan to fine-tune the model multiple times with different settings.
- **device**: Sets the computation device. If you have an Apple Silicon Mac, it utilizes the Metal Performance Shaders (MPS); otherwise, it defaults to the CPU.

**Loading the Datasets**

```python
train_dataset = load_from_disk('train_dataset')
eval_dataset = load_from_disk('eval_dataset')
```

We load the training and evaluation datasets we prepared earlier.

**Loading the Model and Tokenizer**

```python
model_name = '/Users/alexweatherley/Documents/LLMs/Mistral-7B-v0.3'  # Adjust the path to your model
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
```

- **model_name**: Path to the pre-trained model. Make sure to adjust it to where your model is located.
- **tokenizer**: Loads the tokenizer associated with the model.
- **Padding Token**: Ensures the tokenizer has a padding token. If it doesn't, we set it to the end-of-sequence token.
- **model**: Loads the pre-trained causal language model and moves it to the specified device.

**Enabling Gradient Computation on Input Embeddings**

```python
model.enable_input_require_grads()
```

This enables gradient computation on the input embeddings, which is necessary for fine-tuning.

**Applying PEFT with LoRA**

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
```

- **peft_config**: Configures the LoRA settings.
  - **task_type**: Specifies the task; in this case, causal language modeling.
  - **inference_mode**: Set to `False` since we're training, not just inferring.
  - **r, lora_alpha, lora_dropout**: Hyperparameters controlling the adaptation.
- **get_peft_model**: Wraps our model with the PEFT configuration.

**Enabling Gradient Checkpointing**

```python
model.gradient_checkpointing_enable()
```

Enables gradient checkpointing to save memory during training by not storing all intermediate activations.

**Verifying Model Parameters**

```python
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
```

This loop prints out which parameters require gradients, helping us verify that only the intended parts of the model are being updated.

**Tokenization Function**

```python
def tokenize_function(examples):
    inputs = []
    for i in tqdm(range(len(examples['input'])), desc='Tokenizing'):
        prompt = examples['input'][i]
        response = examples['output'][i]
        full_text = prompt + '\n' + response
        inputs.append(full_text)
    
    tokenized_outputs = tokenizer(
        inputs,
        truncation=True,
        max_length=512,  # Increased max_length to 512
        padding='max_length',
    )
    tokenized_outputs['labels'] = tokenized_outputs['input_ids'].copy()
    return tokenized_outputs
```

- **Concatenating Prompt and Response**: For each example, we combine the prompt and the response, separated by a newline.
- **Tokenization**: Tokenizes the combined text with a maximum length of 512 tokens, truncating and padding as necessary.
- **Labels**: Sets the labels for training; since we're doing language modeling, the labels are the same as the input IDs.

**Tokenizing the Datasets**

```python
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['input', 'output'],
    desc='Tokenizing Training Dataset'
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['input', 'output'],
    desc='Tokenizing Evaluation Dataset'
)
```

We apply the tokenization function to both the training and evaluation datasets.

**Initializing the Data Collator**

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

- **Data Collator**: Prepares batches of data during training.
- **mlm=False**: Indicates we're not using masked language modeling but causal language modeling.

**Setting Up Training Arguments**

```python
training_args = TrainingArguments(
    output_dir='./results-' + iteration,
    eval_strategy='steps',
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Adjusted for limited hardware
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.0,
    save_total_limit=2,
    fp16=False,  # Disable 16-bit floating point precision
    bf16=True,   # Enable bfloat16 precision (if supported)
)
```

- **Output Directory**: Where checkpoints and logs will be saved.
- **Evaluation and Saving Strategy**: Evaluates and saves the model every 500 steps.
- **Logging**: Logs training progress every 100 steps.
- **Epochs**: Number of times to iterate over the entire training dataset.
- **Batch Sizes**: Set to 1 due to hardware limitations; gradient accumulation is used to simulate a larger batch size.
- **Learning Rate and Weight Decay**: Standard settings for fine-tuning.
- **Precision**: Disables FP16 and enables BF16 precision to improve performance on supported hardware.

**Initializing the Trainer**

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)
```

The Trainer handles the training loop, evaluation, and saving of checkpoints.

**Starting the Training Process**

```python
trainer.train()
```

Begins fine-tuning the model with our custom dataset.

**Saving the Fine-Tuned Model**

```python
trainer.save_model('./fine-tune-' + iteration)
tokenizer.save_pretrained('./fine-tune-' + iteration)
```

After training, we save the fine-tuned model and tokenizer for future use.

**Notes and Considerations**

- **Hardware Limitations**: Fine-tuning large models can be resource-intensive. Adjusting batch sizes and using gradient accumulation helps mitigate this.
- **Training Time**: Be prepared for the training process to take some time, especially on a laptop.
- **Model Path**: Ensure that `model_name` points to the correct location of your pre-trained model.
- **BF16 Precision**: Bfloat16 precision can improve training speed and reduce memory usage if your hardware supports it.

**Wrapping Up**

By executing this script, we're customizing the language model to generate descriptions of cities in the style of *Invisible Cities*. With the fine-tuned model, you can prompt it to create new, whimsical cityscapes that perhaps even Kublai Khan would find intriguing.

Now, it's time to let the model train and see what new cities it can imagine. Happy fine-tuning!

Let's get training!

And because I'm a little paranoid about my laptop going to sleep and messing things up. In a separate terminal I grabbed the process id of our training run and ensured wakefulness while the fine-tuning goes on:

```shell
/your_project/ % uv run fine_tune_model_peft.py
```

```shell
% ps -a

  PID TTY           TIME CMD
42892 ttys001    0:00.23 /bin/zsh -il
83642 ttys001    0:00.02 uv run fine_tune_model_peft.py
42984 ttys003    0:00.08 /bin/zsh -i

% caffeinate -w 42892
```
If it's not too late in your day, you could have a coffee too.

*gif of spinning egg-timer*

  

TODO clone llama.cpp

  

from /llama.cpp/

llama.cpp % python3 -m venv .venv

llama.cpp % source .venv/bin/activate

llama.cpp % pip install -r requirements.txt

  
  

```

python3 convert_lora_to_gguf.py ~/Development/infinite_cities/fine-tuned-model --base ~/Documents/LLMs/Mistral-7B-v0.3 --outfile ~/Documents/LLMs/mistral7B-v0.3-inf-cities-fp16.gguf

...

Writing: 100%|███████████████████████████| 6.82M/6.82M [00:00<00:00, 889Mbyte/s]

  

INFO:lora-to-gguf:Model successfully exported to /Users/alexweatherley/Documents/LLMs/mistral7B-v0.3-inf-cities-02-fp16.gguf

```

llama.cpp % deactivate

cd to LLMs folder

  

Import to ollama. Create a file called 'Modelfile' with paths to both the base model and fine-tune. The base path in FROM, the fine-tune path in ADAPTER

```

FROM /path/to/Mistral-7B-v0.3

ADAPTER /path/to/mistral7B-v0.3-inf-cities-fp16.gguf

```

## Importing our model

To import our model into ollama

```console
% ollama create infinite-cities-002

transferring model data ⠦
```

## Iterations and afterthoughts - *token lengths, CAUSAL_LM vs SEQ_2_SEQ*