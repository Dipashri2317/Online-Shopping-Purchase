from pathlib import Path

import pandas as pd

from ucimlrepo import fetch_ucirepo


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = BASE_DIR / "online_shoppers_intention.csv"


def main():

    print(
        "Downloading UCI Online Shoppers "
        "Purchasing Intention Dataset..."
    )

    dataset = fetch_ucirepo(id=468)

    features = dataset.data.features

    targets = dataset.data.targets

    data = pd.concat(
        [features, targets],
        axis=1
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved {len(data):,} rows to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()