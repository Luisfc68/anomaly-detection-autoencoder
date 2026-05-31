from fraud.config import FIGURES_DIR, set_seed
from fraud.data import load_raw
from fraud.eda import plot_class_balance, plot_amount_by_class, plot_time_of_day, plot_amount, summarize

def main():
    set_seed()
    df = load_raw()
    summarize(df)
    plot_class_balance(df)
    plot_amount(df)
    plot_amount_by_class(df)
    plot_time_of_day(df)
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()