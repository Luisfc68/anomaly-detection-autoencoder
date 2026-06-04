from fraud.config import FIGURES_DIR, set_seed
from fraud.data import ensure_dataset, load_raw
from fraud.eda import (
    plot_amount,
    plot_amount_by_class,
    plot_class_balance,
    plot_time_of_day,
    summarize,
)


def main():
    set_seed()
    ensure_dataset()
    df = load_raw()
    summarize(df)
    plot_class_balance(df)
    plot_amount(df)
    plot_amount_by_class(df)
    plot_time_of_day(df)
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
