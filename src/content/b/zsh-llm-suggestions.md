---
title: Local LLM enabled Zsh shell command suggester/explainer
publishDate: 2025-09-27
tags: ["Tech", "LLMs", "AI"]
---

### A Zsh Plugin for Local and Cloud AI Command Suggestions

This post introduces an updated fork of the `zsh-llm-suggestions` plugin, which integrates Large Language Models (LLMs) into the Zsh command line. The primary goal of this tool is to reduce the need to switch contexts to a web browser for looking up shell command syntax.

The original project provided a solid foundation, and this fork adds support for modern, local-first model runners and improves the dependency management process.

#### Core Functionality

The plugin operates in two modes:

1. **Command Generation**: A user types a descriptive prompt (e.g., "uncompress a .tar.gz file") and uses a hotkey. The plugin sends this prompt to a configured LLM and replaces the text with the suggested command.
    
2. **Command Explanation**: For an existing command on the line, a hotkey can be used to request an explanation from the LLM, which is then printed to the terminal.
    

This functionality is demonstrated in the following recording:

#### Key Features of This Fork

This version introduces several key updates:

- **MLX Support**: Enables running local models efficiently on Apple Silicon via Apple's [MLX](https://opensource.apple.com/projects/mlx/) framework. This is ideal for offline use and privacy.
    
- **Ollama Support**: Integrates with [Ollama](https://ollama.com/) for cross-platform local model execution, supporting a wide range of open-source models on macOS, Linux, and Windows.
    
- **UV for Dependency Management**: Uses Astral's `uv` for fast and self-contained Python package installation, simplifying the setup process.
    

#### Basic Setup Guide

Here are the essential steps to get the plugin running.

**1. Clone the repository:**

Shell

```
git clone https://github.com/64MM4-KN1F3/zsh-llm-suggestions.git ~/.zsh-plugins/zsh-llm-suggestions
```

**2. Configure your `.zshrc`:** Add the following configuration to your `~/.zshrc` file. This example uses `uv` and is configured for local models with MLX and Ollama.

Shell

```
# --- Zsh LLM Suggestions Configuration ---

# Use UV for automatic dependency management
export ZSH_LLM_SUGGESTIONS_USE_UV=true

# Specify local model identifiers
export ZSH_LLM_SUGGESTIONS_MLX_MODEL="mlx-community/Phi-3-mini-4k-instruct-8bit"
export ZSH_LLM_SUGGESTIONS_OLLAMA_MODEL="llama3"

# Source the plugin script
source ~/.zsh-plugins/zsh-llm-suggestions/zsh-llm-suggestions.zsh

# Bind functions to hotkeys
bindkey '^L' zsh_llm_suggestions_mlx         # Suggest with MLX
bindkey '^[l' zsh_llm_suggestions_mlx_explain # Explain with MLX
```

_The [README.md](https://github.com/64MM4-KN1F3/zsh-llm-suggestions) on GitHub contains the complete documentation, including setup for cloud providers like OpenAI and GitHub Copilot._

**3. Reload your shell:** To apply the changes, restart your terminal or run `exec zsh`.

#### A Note on Usage

LLMs can produce incorrect or even destructive commands. It is important to always review a command suggestion for accuracy and safety before executing it.

The project is available on GitHub for those who find this utility useful for their command-line workflow. You can find all the code and full documentation there.