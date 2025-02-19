import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

# Dataset configurations
DATASETS = [
    ("breast_cancer", load_breast_cancer(as_frame=True)),
    ("parkinsons", fetch_openml("parkinsons", as_frame=True)),
    ("heart_disease", fetch_openml("heart-statlog", version=1, as_frame=True)),
    ("diabetes", fetch_openml("diabetes", version=1, as_frame=True)),
]

# Load and split each dataset
for name, data in DATASETS:
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"tabpfn/{name}_train.csv", index=False)
    test_df.to_csv(f"tabpfn/{name}_test.csv", index=False)
