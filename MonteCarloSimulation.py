import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from typing import List, Dict
import sys

# Constants
DEFAULT_CONFIG_PATH = 'config.json'
NUM_SAMPLE_PATHS = 5
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Set a consistent theme for plots
sns.set_theme(style='whitegrid')

# Configure logging with multiple levels
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class MonteCarloTradingSimulation:
    """
    A Monte Carlo simulation for trading balance progression.

    Attributes:
        starting_balance (float): The initial balance of the trading account.
        win_rate (float): The probability of winning a trade.
        average_gain (float): The average gain from a winning trade.
        average_loss (float): The average loss from a losing trade.
        num_simulations (int): The number of simulations to run.
        num_trades (int): The number of trades to simulate.
        ending_balances (np.ndarray): The ending balances from each simulation.
        sample_paths (np.ndarray): The balance progression of sample simulations.
        max_drawdowns (np.ndarray): The maximum drawdowns from each simulation.
        max_wins_in_a_row (np.ndarray): The maximum consecutive wins from each simulation.
        max_losses_in_a_row (np.ndarray): The maximum consecutive losses from each simulation.
    """

    def __init__(
            self,
            starting_balance: float,
            win_rate: float,
            average_gain: float,
            average_loss: float,
            num_simulations: int,
            num_trades: int
    ):
        try:
            self._validate_parameters(starting_balance, win_rate, average_gain, average_loss)
        except ValueError as e:
            logger.error(f"Parameter validation failed: {e}")
            raise

        self.starting_balance = starting_balance
        self.win_rate = win_rate
        self.average_gain = average_gain
        self.average_loss = average_loss
        self.num_simulations = num_simulations
        self.num_trades = num_trades
        self.ending_balances = np.zeros(self.num_simulations)
        self.sample_paths = np.zeros((NUM_SAMPLE_PATHS, self.num_trades + 1))
        self.max_drawdowns = np.zeros(self.num_simulations)
        self.max_wins_in_a_row = np.zeros(self.num_simulations)
        self.max_losses_in_a_row = np.zeros(self.num_simulations)

    def _validate_parameters(
            self,
            starting_balance: float,
            win_rate: float,
            average_gain: float,
            average_loss: float
    ):
        """
        Validate the input parameters for the simulation.

        Raises:
            ValueError: If any parameter is out of the expected range.
        """
        if not (0 <= win_rate <= 1):
            raise ValueError("Win rate must be between 0 and 1.")
        if any(param <= 0 for param in [starting_balance, average_gain, average_loss]):
            raise ValueError("Starting balance, average gain, and average loss must be positive.")

    def run_simulation(self):
        """
        Run the Monte Carlo simulation for trading balance progression.
        """
        try:
            logger.debug("Starting simulation run.")
            outcomes = np.random.rand(self.num_simulations, self.num_trades) < self.win_rate
            gains_losses = np.where(outcomes, self.average_gain, -self.average_loss)
            balance_histories = np.cumsum(gains_losses, axis=1) + self.starting_balance
            self.ending_balances = balance_histories[:, -1]
            self.sample_paths[:, :] = np.hstack((
                np.full((NUM_SAMPLE_PATHS, 1), self.starting_balance),
                balance_histories[:NUM_SAMPLE_PATHS]
            ))
            peak_balances = np.maximum.accumulate(balance_histories, axis=1)
            drawdowns = (peak_balances - balance_histories) / peak_balances
            self.max_drawdowns = np.max(drawdowns, axis=1)
            self.max_wins_in_a_row = self._calculate_max_streak(outcomes, win=True)
            self.max_losses_in_a_row = self._calculate_max_streak(outcomes, win=False)
            logger.debug("Simulation run completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during simulation run: {e}")
            raise

    def _calculate_max_streak(self, outcomes: np.ndarray, win: bool) -> np.ndarray:
        """
        Calculate the maximum consecutive wins or losses from the outcomes.

        Args:
            outcomes (np.ndarray): A boolean array of trade outcomes.
            win (bool): Whether to calculate the maximum wins or losses.

        Returns:
            np.ndarray: The maximum consecutive wins or losses.
        """
        streaks = np.where(outcomes, 1, 0) if win else np.where(~outcomes, 1, 0)
        max_streaks = np.array([self._max_consecutive(x) for x in streaks])
        return max_streaks

    @staticmethod
    def _max_consecutive(arr: np.ndarray) -> int:
        """
        Calculate the maximum consecutive occurrences of a value in an array.

        Args:
            arr (np.ndarray): The input array.

        Returns:
            int: The maximum consecutive occurrences of a value in the array.
        """
        return max(
            (len(seq) for seq in np.split(arr, np.where(arr == 0)[0]) if len(seq) > 0),
            default=0
        )

    def display_statistics(self):
        """
        Display the statistics of the Monte Carlo simulation.
        """
        try:
            logger.debug("Calculating statistics.")
            mean_balance = np.mean(self.ending_balances)
            std_dev_balance = np.std(self.ending_balances)
            min_balance = np.min(self.ending_balances)
            max_balance = np.max(self.ending_balances)
            var_95 = np.percentile(self.ending_balances, 5)
            cvar_95 = np.mean(self.ending_balances[self.ending_balances <= var_95])
            daily_return = (self.ending_balances / self.starting_balance - 1) / self.num_trades
            avg_percentage_gain_loss_per_day = np.mean(daily_return) * 100
            sharpe_ratio = (np.mean(daily_return) / np.std(daily_return)) * np.sqrt(252)
            downside_risk = np.std(daily_return[daily_return < 0])
            sortino_ratio = (np.mean(daily_return) / downside_risk * np.sqrt(252)) if downside_risk != 0 else 0
            avg_max_drawdown = np.mean(self.max_drawdowns)
            avg_max_wins_in_a_row = np.mean(self.max_wins_in_a_row)
            avg_max_losses_in_a_row = np.mean(self.max_losses_in_a_row)

            logger.info("Monte Carlo Simulation Results:")
            logger.info(f"Mean Ending Balance: ${mean_balance:,.2f}")
            logger.info(f"Standard Deviation of Ending Balance: ${std_dev_balance:,.2f}")
            logger.info(f"Minimum Ending Balance: ${min_balance:,.2f}")
            logger.info(f"Maximum Ending Balance: ${max_balance:,.2f}")
            logger.info(f"Value at Risk (5%): ${var_95:,.2f}")
            logger.info(f"Conditional Value at Risk (5%): ${cvar_95:,.2f}")
            logger.info(f"Average Percentage Gain/Loss per Day: {avg_percentage_gain_loss_per_day:.2f}%")
            logger.info(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Annualized Sortino Ratio: {sortino_ratio:.2f}")
            logger.info(f"Average Maximum Drawdown: {avg_max_drawdown:.2%}")
            logger.info(f"Average Number of Wins in a Row: {avg_max_wins_in_a_row:.2f}")
            logger.info(f"Average Number of Losses in a Row: {avg_max_losses_in_a_row:.2f}")
            logger.debug("Statistics calculation completed.")
        except Exception as e:
            logger.error(f"An error occurred during statistics calculation: {e}")
            raise

    def plot_results(self):
        """
        Plot the results of the Monte Carlo simulation.
        """
        try:
            logger.debug("Starting to plot results.")
            self._plot_histogram()
            self._plot_density()
            self._plot_sample_paths()
            self._plot_cdf()
            logger.debug("Plotting completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during plotting: {e}")
            raise

    def _plot_histogram(self):
        """
        Plot a histogram of the ending balances.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.ending_balances, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Ending Balances')
        plt.xlabel('Ending Balance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def _plot_density(self):
        """
        Plot a density plot of the ending balances.
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.ending_balances, color="purple", fill=True)
        plt.axvline(np.mean(self.ending_balances), color='red', linestyle='--', label='Mean')
        plt.axvline(np.median(self.ending_balances), color='blue', linestyle='--', label='Median')
        plt.title('Density Plot of Ending Balances')
        plt.xlabel('Ending Balance')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_sample_paths(self):
        """
        Plot the balance progression of sample simulations.
        """
        plt.figure(figsize=(10, 6))
        for i, path in enumerate(self.sample_paths):
            plt.plot(path, label=f'Simulation {i + 1}')
        plt.title('Balance Progression Over Trades in Sample Simulations')
        plt.xlabel('Number of Trades')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_cdf(self):
        """
        Plot the cumulative distribution function of the ending balances.
        """
        sorted_balances = np.sort(self.ending_balances)
        cumulative_prob = np.arange(1, len(sorted_balances) + 1) / len(sorted_balances)
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_balances, cumulative_prob, color='darkgreen')
        plt.title('Cumulative Distribution Function of Ending Balances')
        plt.xlabel('Ending Balance')
        plt.ylabel('Cumulative Probability')
        plt.grid(True)
        plt.show()


def load_configuration(config_path: str) -> Dict:
    """
    Load simulation configuration from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        Dict: The configuration parameters.

    Raises:
        ConfigurationError: If the configuration file is missing or invalid.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            logger.debug(f"Configuration loaded from {config_path}.")
            return config
    except FileNotFoundError:
        logger.critical(f"Configuration file {config_path} not found.")
        raise ConfigurationError(f"Configuration file {config_path} not found.")
    except json.JSONDecodeError as e:
        logger.critical(f"Error decoding JSON from {config_path}: {e}")
        raise ConfigurationError(f"Error decoding JSON from {config_path}: {e}")


def main(config_path: str = DEFAULT_CONFIG_PATH):
    """
    Main function to execute the Monte Carlo trading simulation.

    Args:
        config_path (str): Path to the JSON configuration file.
    """
    try:
        config = load_configuration(config_path)
        simulation = MonteCarloTradingSimulation(
            starting_balance=config["starting_balance"],
            win_rate=config["win_rate"],
            average_gain=config["average_gain"],
            average_loss=config["average_loss"],
            num_simulations=config["num_simulations"],
            num_trades=config["num_trades"]
        )
        simulation.run_simulation()
        simulation.display_statistics()
        simulation.plot_results()
    except ConfigurationError as ce:
        logger.critical(f"Configuration Error: {ce}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()