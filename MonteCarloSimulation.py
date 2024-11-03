import traceback
from functools import wraps
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Tuple
import sys
from Logger import Logger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

DEFAULT_CONFIG_PATH = 'config.json'
NUM_SAMPLE_PATHS = 20
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

sns.set_theme(style='whitegrid')


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        args[0].logger.info(f"Execution time for {func.__name__}: {exec_time:.4f} seconds")
        return result

    return wrapper


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
            num_trades: int,
            logger: Logger,
            results_dir: str = None,
            figures_dir: str = None
    ):
        self.logger = logger
        try:
            self._validate_parameters(starting_balance, win_rate, average_gain, average_loss)
        except ValueError as e:
            self.logger.error(f"Parameter validation failed: {e}")
            raise

        self.num_trading_days = 252
        self.starting_balance = starting_balance
        self.win_rate = win_rate
        self.average_gain = average_gain
        self.average_loss = average_loss
        self.num_simulations = num_simulations
        self.num_trades = num_trades
        self.ending_balances = np.zeros(self.num_simulations)
        self.median_ending_balance = 0
        self.sample_paths = np.zeros((NUM_SAMPLE_PATHS, self.num_trades + 1))
        self.max_drawdowns = np.zeros(self.num_simulations)
        self.max_wins_in_a_row = np.zeros(self.num_simulations)
        self.max_losses_in_a_row = np.zeros(self.num_simulations)
        if results_dir is None or figures_dir is None:
            self.results_dir, self.figures_dir = self._create_results_directory()
        else:
            self.results_dir = results_dir
            self.figures_dir = figures_dir

    def _validate_parameters(
            self,
            starting_balance: float,
            win_rate: float,
            average_gain: float,
            average_loss: float
    ) -> None:
        """
        Validate the input parameters for the simulation.

        Raises:
            ValueError: If any parameter is out of the expected range.
        """
        if not (0 <= win_rate <= 1):
            raise ValueError("Win rate must be between 0 and 1.")
        if any(param <= 0 for param in [starting_balance, average_gain, average_loss]):
            raise ValueError("Starting balance, average gain, and average loss must be positive.")

    def run_simulation(self) -> None:
        """
        Run the Monte Carlo simulation for trading balance progression.
        """
        try:
            self.logger.info("Starting simulation run.")

            outcomes = np.random.rand(self.num_simulations, self.num_trades) < self.win_rate
            self.logger.info("Outcomes generated for simulations")

            gains_losses = np.where(outcomes, self.average_gain, -self.average_loss)
            self.logger.info("Gains and losses calculated")

            balance_histories = np.cumsum(gains_losses, axis=1) + self.starting_balance
            self.logger.info("Cumulative balance histories computed")

            self.ending_balances = balance_histories[:, -1]
            self.logger.info("Ending balances determined")

            sample_indices = np.random.choice(self.num_simulations, NUM_SAMPLE_PATHS, replace=False)
            self.sample_paths[:, :] = np.hstack((
                np.full((NUM_SAMPLE_PATHS, 1), self.starting_balance), balance_histories[sample_indices]
            ))
            self.logger.info("Sample paths generated")

            peak_balances = np.maximum.accumulate(balance_histories, axis=1)
            self.logger.info("Peak balances calculated")

            drawdowns = (peak_balances - balance_histories) / peak_balances
            self.logger.info("Drawdowns computed")

            self.max_drawdowns = np.max(drawdowns, axis=1)
            self.logger.info("Max drawdowns identified")

            self.max_wins_in_a_row = self._calculate_max_streak(outcomes, win=True)
            self.logger.info("Max wins in a row calculated")

            self.max_losses_in_a_row = self._calculate_max_streak(outcomes, win=False)
            self.logger.info("Max losses in a row calculated")

            self.logger.info("Simulation run completed successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred during simulation run: {e}")
            raise

    def _create_results_directory(self) -> Tuple[str, str]:
        """
        Create a directory to store the simulation results and figures.
        :return: A tuple containing the path to the results directory and the figures directory.
        """
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = Path("statistics") / current_time
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        return str(results_dir), str(figures_dir)

    @measure_time
    def _calculate_max_streak(self, outcomes: np.ndarray, win: bool) -> np.ndarray:
        """
        Calculate the maximum consecutive wins or losses from the outcomes in parallel.

        Args:
            outcomes (np.ndarray): A boolean array of trade outcomes.
            win (bool): Whether to calculate the maximum wins or losses.

        Returns:
            np.ndarray: The maximum consecutive wins or losses.
        """
        streaks = np.where(outcomes, 1, 0) if win else np.where(~outcomes, 1, 0)
        with ProcessPoolExecutor() as executor:
            max_streaks = np.array(list(executor.map(self._max_consecutive, streaks)))
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

    def display_statistics(self) -> None:
        """
        Display the statistics of the Monte Carlo simulation.
        """
        try:
            self.logger.info("Calculating statistics.")
            mean_balance = np.mean(self.ending_balances)
            self.median_ending_balance = np.median(self.ending_balances)
            std_dev_balance = np.std(self.ending_balances)
            min_balance = np.min(self.ending_balances)
            max_balance = np.max(self.ending_balances)
            var_95 = np.percentile(self.ending_balances, 5)
            cvar_95 = np.mean(self.ending_balances[self.ending_balances <= var_95])
            daily_return = (self.ending_balances / self.starting_balance - 1) / self.num_trading_days
            avg_percentage_gain_loss_per_day = np.mean(daily_return) * 100

            # Daily risk-free rate, risk-free rate was calculated from 10 year treasury bond rate
            # And inflation for the September 2024 in the USA
            risk_free_rate = 0.0184 / 252
            sharpe_ratio = ((np.mean(daily_return) - risk_free_rate) / np.std(daily_return)) * np.sqrt(252)
            downside_risk = np.std(daily_return[daily_return < 0])
            sortino_ratio = ((np.mean(daily_return) - risk_free_rate) / downside_risk * np.sqrt(
                252)) if downside_risk != 0 else 0
            avg_max_drawdown = np.mean(self.max_drawdowns)
            avg_max_wins_in_a_row = np.mean(self.max_wins_in_a_row)
            avg_max_losses_in_a_row = np.mean(self.max_losses_in_a_row)

            # gain-to-pain ratio
            gains = self.ending_balances[self.ending_balances > self.starting_balance] - self.starting_balance
            losses = self.starting_balance - self.ending_balances[self.ending_balances < self.starting_balance]
            gain_to_pain_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) != 0 else np.inf

            self.logger.info("Statistics calculated.")

            stats_file_path = Path(self.results_dir) / "statistics.md"
            self.logger.info(f"Writing statistics to \"{stats_file_path}\".")

            with open(stats_file_path, 'w') as f:
                f.write(f"## Monte Carlo Simulation Results:\n\n")
                f.write(f"- Mean Ending Balance: ${mean_balance:,.2f}\n")
                f.write(f"- Median Ending Balance: ${self.median_ending_balance:,.2f}\n")
                f.write(f"- Standard Deviation of Ending Balance: ${std_dev_balance:,.2f}\n")
                f.write(f"- Minimum Ending Balance: ${min_balance:,.2f}\n")
                f.write(f"- Maximum Ending Balance: ${max_balance:,.2f}\n")
                f.write(f"- Average Percentage Gain/Loss per Day: {avg_percentage_gain_loss_per_day:.2f}%\n")
                f.write(f"- Average Maximum Drawdown: {avg_max_drawdown:.2%}\n\n")
                f.write(f"- Average Number of Wins in a Row: {avg_max_wins_in_a_row:.2f}\n")
                f.write(f"- Average Number of Losses in a Row: {avg_max_losses_in_a_row:.2f}\n\n")
                f.write(f"#### Advanced Statistics:\n\n")
                f.write(f"- Value at Risk (5%): ${var_95:,.2f}\n")
                f.write(f"- Conditional Value at Risk (5%): ${cvar_95:,.2f}\n")
                f.write(f"- Annualized Sharpe Ratio: {sharpe_ratio:.2f}\n")
                f.write(f"- Annualized Sortino Ratio: {sortino_ratio:.2f}\n")
                f.write(f"- Gain-to-Pain Ratio: {gain_to_pain_ratio:.2f}\n")

                f.write("Statistics calculation completed.\n")

            self.logger.info(f"Statistics calculation completed. Results saved to \"{stats_file_path}\".")
        except Exception as e:
            self.logger.error(f"An error occurred during statistics calculation: {e}")
            raise

    def plot_results(self) -> None:
        """
        Plot the results of the Monte Carlo simulation.
        """
        try:
            self.logger.info("Starting to plot results.")
            self._plot_histogram()
            self._plot_density()
            self._plot_sample_paths()
            self._plot_cdf()
            self.logger.info("Plotting completed successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred during plotting: {e}")
            raise

    def _plot_histogram(self) -> None:
        """
        Plot and save a histogram of the ending balances.
        """
        self.logger.info("Plotting histogram of ending balances.")
        plt.figure(figsize=(10, 6))
        plt.hist(self.ending_balances, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Ending Balances')
        plt.xlabel('Ending Balance')
        plt.ylabel('Frequency')
        plt.grid(True)
        self.logger.info("Saving histogram to figures directory.")
        plt.savefig(Path(self.figures_dir) / "ending_balances_histogram.png")
        plt.close()
        self.logger.info(f"\"ending_balances_histogram\" plot saved to \"{self.figures_dir}\".")

    def _plot_density(self) -> None:
        """
        Plot and save a density plot of the ending balances.
        """
        self.logger.info("Plotting density plot of ending balances.")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.ending_balances, color="purple", fill=True)
        plt.axvline(np.mean(self.ending_balances), color='red', linestyle='--', label='Mean')
        plt.axvline(np.median(self.ending_balances), color='blue', linestyle='--', label='Median')
        plt.title('Density Plot of Ending Balances')
        plt.xlabel('Ending Balance')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        self.logger.info("Saving density plot to figures directory.")
        plt.savefig(Path(self.figures_dir) / "ending_balances_density.png")
        plt.close()
        self.logger.info(f"\"ending_balances_density\" plot saved to \"{self.figures_dir}\".")

    def _plot_sample_paths(self) -> None:
        """
        Plot and save the balance progression of sample simulations.
        """
        self.logger.info("Plotting sample paths.")
        plt.figure(figsize=(10, 6))
        for i, path in enumerate(self.sample_paths):
            plt.plot(path, label=f'Simulation {i + 1}')
        plt.title('Balance Progression Over Trades in Sample Simulations')
        plt.xlabel('Number of Trades')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True)
        self.logger.info("Saving sample paths plot to figures directory.")
        plt.savefig(Path(self.figures_dir) / "sample_paths.png")
        plt.close()
        self.logger.info(f"\"Sample_paths\" plot saved to \"{self.figures_dir}\".")

    def _plot_cdf(self) -> None:
        """
        Plot and save the cumulative distribution function of the ending balances.
        """
        self.logger.info("Plotting CDF of ending balances.")
        sorted_balances = np.sort(self.ending_balances)
        cumulative_prob = np.arange(1, len(sorted_balances) + 1) / len(sorted_balances)
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_balances, cumulative_prob, color='darkgreen')
        plt.title('Cumulative Distribution Function of Ending Balances')
        plt.xlabel('Ending Balance')
        plt.ylabel('Cumulative Probability')
        plt.grid(True)
        self.logger.info("Saving CDF plot to figures directory.")
        plt.savefig(Path(self.figures_dir) / "ending_balances_cdf.png")
        plt.close()
        self.logger.info(f"\"ending_balances_cdf\" plot saved to \"{self.figures_dir}\".")


def load_configuration(config_path: str, logger) -> Dict:
    """
    Load simulation configuration from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file.
        :param config_path: The path to the JSON configuration file.
        :param logger: The logger instance.

    Returns:
        Dict: The configuration parameters.

    Raises:
        ConfigurationError: If the configuration file is missing or invalid.
        json.JSONDecodeError: If the JSON file cannot be decoded.

    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            logger.info(f"Configuration loaded from {config_path}.")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found.")
        raise ConfigurationError(f"Configuration file {config_path} not found.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path}: {e}")
        raise ConfigurationError(f"Error decoding JSON from {config_path}: {e}")


def main(config_path: str = DEFAULT_CONFIG_PATH):
    """
    Main function to execute the Monte Carlo trading simulation.

    Args:
        config_path (str): Path to the JSON configuration file.
    """
    logger = Logger(name=__name__, log_file=f"{__name__}.log").get_logger()

    try:
        config = load_configuration(config_path, logger)
        simulation = MonteCarloTradingSimulation(
            starting_balance=config["starting_balance"],
            win_rate=config["win_rate"],
            average_gain=config["average_gain"],
            average_loss=config["average_loss"],
            num_simulations=config["num_simulations"],
            num_trades=config["num_trades"],
            logger=logger
        )
        simulation.run_simulation()
        simulation.display_statistics()
        simulation.plot_results()
        logger.info("Simulation completed successfully.")
    except ConfigurationError as ce:
        logger.critical(f"Configuration Error: {ce}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    except ValueError as ve:
        logger.critical(f"Value Error: {ve}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    except FileNotFoundError as fnfe:
        logger.critical(f"File Not Found Error: {fnfe}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    except KeyError as ke:
        logger.critical(f"Key Error: {ke}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    except TypeError as te:
        logger.critical(f"Type Error: {te}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
