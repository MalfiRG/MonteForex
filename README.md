# Monte Carlo Trading Simulation Script

## Introduction

This project provides a **Monte Carlo Simulator** for evaluating the profitability of **Forex trading strategies**. By using statistical tools and visualizations, traders can make more informed decisions and adjust their strategies to maximize profitability.

## Table of Contents

- [Introduction](#introduction)
- [Detailed Project Description](#detailed-project-description)
- [Prerequisites](#prerequisites)
- [Releases](#releases)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up a Virtual Environment (Optional)](#2-set-up-a-virtual-environment-optional)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Configure the Simulation](#4-configure-the-simulation)
  - [5. Run the Simulation](#5-run-the-simulation)
  - [6. View Results](#6-view-results)
  - [7. Compile to Executable](#7-compile-to-executable)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing Guidelines](#contributing-guidelines)
- [License Information](#license-information)
- [Contact Information](#contact-information)

## Detailed Project Description

The **Monte Carlo Trading Simulation Script** allows traders to simulate various trading scenarios and estimate potential outcomes based on different Forex strategies. By adjusting parameters such as starting balance, win rate, average gain, and average loss, users can visualize how these factors impact overall profitability.

## Prerequisites

Ensure you have the following installed on your system:

- **Python 3.6 or higher**
- **pip** (Python package installer)

## Releases

Can be found [here](https://github.com/MalfiRG/MonteForex/releases).

## Step-by-Step Guide

### 1. Clone the Repository

Clone the repository to your local machine:

```sh
git clone https://github.com/MalfiRG/MonteForex.git
cd MonteForex
```

### 2. Set Up a Virtual Environment (Optional)

It is recommended to set up a virtual environment to manage dependencies:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages listed in the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### 4. Configure the Simulation

Edit the `config.json` file to set your desired simulation parameters. Below is an example configuration:

```json
{
    "starting_balance": 56000,
    "win_rate": 0.5,
    "average_gain": 600,
    "average_loss": 380,
    "num_simulations": 10000,
    "num_trades": 3000
}
```

### 5. Run the Simulation

Run the simulation by executing the script:

```sh
python MonteCarloSimulation.py
```

### 6. View Results

The script will display statistics and generate plots of the simulation results. The results are saved in the 'statistics' directory. Each run produces a folder in the format YYYY-MM-DD_HH-MM-SS containing the following files:
1. statistics.md
2. Figures:
    - ending_balances_cfd.png
    - ending_balances_density.png
    - ending_balances_histogram.png
    - sample_paths.png

### 7. Compile to Executable

To compile the script into an executable using PyInstaller, follow these steps:
1. Install PyInstaller:
```sh
pip install pyinstaller
```

2. Compile the script:
```sh
pyinstaller --onefile MonteCarloSimulation.py
```

**WARNING**: Some antivirus engines might flag the executable as a false positive. To avoid this:
##### 1. Use a Locally Compiled Bootloader
You can try compiling PyInstaller from source to create a custom bootloader, which may help avoid detection:

Clone the PyInstaller repository:

```shell
git clone https://github.com/pyinstaller/pyinstaller.git
cd pyinstaller/bootloader
```

Build the bootloader:

```sh
python ./waf all
```

Use this custom-built version of PyInstaller to generate your executable.
This method reduces the likelihood of your executable being flagged since the bootloader will be unique and not widely recognized by antivirus software.

##### 2. Digitally Sign Your Executable
You can obtain a code-signing certificate from a Certificate Authority (CA) to sign your executable. This will help establish trust and reduce the likelihood of false positives.
##### 3. Submit the Executable for Whitelisting
If your executable is being flagged by antivirus software, you can submit it to the respective antivirus vendor for whitelisting. 
##### 4. Whitelist the File Locally
Self-explanatory, you can whitelist the file in your antivirus software.

### 8. Place config file in the same directory as the executable

It is important to place the config file in the same directory as the executable. This is because the executable will look for the config file in the same directory as the executable.
**Otherwise, the executable will not be able to find the config file and will not run.**

## Usage Examples

Here are some example commands and expected outputs when using this project:

```sh
python MonteCarloSimulation.py

Simulation complete. Results saved to results/output.png.
```

## Troubleshooting

Here are some common issues and solutions:

- **Configuration Errors**: Ensure that `config.json` is correctly formatted and contains all required fields.
- **Dependency Issues**: Verify that all dependencies are installed by running `pip install -r requirements.txt`.

## Contact Information

For questions or discussions about this project, feel free to reach out via email at [malfiraggraclan@gmail.com](mailto:malfiraggraclan@gmail.com) or open a discussion on our GitHub Discussions page.
