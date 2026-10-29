"""MAIN CODE. Run the experiments. Data are downloaded (or read from cache) once."""
import argparse

from data_loader import load_market_data
from market_analysis import run_empirical_analysis
from semi_synthetic_sim import run_semi_synthetic_monte_carlo
from simulation import run_monte_carlo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true",
                   help="only the synthetic Monte Carlo (no market data needed)")
    p.add_argument("--refresh", action="store_true", help="re-download the market data")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    plot = not args.no_plots

    run_monte_carlo(plot=plot)
    if not args.offline:
        log_rets = load_market_data(refresh=args.refresh)
        run_empirical_analysis(log_rets, plot=plot)
        run_semi_synthetic_monte_carlo(log_rets, plot=plot)


if __name__ == "__main__":
    main()
