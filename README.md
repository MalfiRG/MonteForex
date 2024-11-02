# How to Set Up the Monte Carlo Trading Simulation Script

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.6 or higher
- `pip` (Python package installer)

## Step-by-Step Guide

### 1. Clone the Repository

Clone the repository containing the Monte Carlo Trading Simulation script to your local machine.

```sh
git clone https://github.com/MalfiRG/MonteForex.git
cd MonteForex
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

Create and activate a virtual environment to manage dependencies.

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages listed in the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

### 4. Configure the Simulation

Edit the `config.json` file to set your desired simulation parameters. The file should look like this:

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

Execute the `MonteCarloSimulation.py` script to run the simulation.

```sh
python MonteCarloSimulation.py
```

### 6. View Results

The script will display statistics and generate plots of the simulation results. Ensure you have a graphical interface available to view the plots.

## Troubleshooting

- **Configuration Errors**: Ensure the `config.json` file is correctly formatted and all required fields are present.
- **Dependency Issues**: Verify that all dependencies are installed correctly by checking the output of `pip install -r requirements.txt`.
