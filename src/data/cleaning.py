import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean merged Walmart sales dataset for EDA and modeling.
    """

    data = data.copy()

    # Convert Date column
    data["Date"] = pd.to_datetime(data["Date"])

    # Fill markdown columns
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

    for col in markdown_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    # Forward-fill slow-changing external variables
    fill_cols = ["CPI", "Unemployment", "Temperature", "Fuel_Price"]

    for col in fill_cols:
        if col in data.columns:
            data[col] = data[col].ffill()

    # Fix target column type
    if "Weekly_Sales" in data.columns:
        data["Weekly_Sales"] = pd.to_numeric(data["Weekly_Sales"], errors="coerce")

    # Remove duplicates
    data = data.drop_duplicates()

    # Sort for time-series consistency
    sort_cols = [col for col in ["Store", "Dept", "Date"] if col in data.columns]
    data = data.sort_values(sort_cols)

    return data


def validate_clean_data(data: pd.DataFrame) -> None:
    """
    Validation checks after cleaning.

    Weekly_Sales should exist for training rows and remain missing
    for test rows because test rows are used for future prediction.
    """

    assert data.duplicated().sum() == 0, "Duplicate rows still exist."
    assert data["Date"].isnull().sum() == 0, "Date column has missing values."

    required_cols = ["Store", "Dept", "Date", "IsHoliday", "is_train", "Weekly_Sales"]

    for col in required_cols:
        assert col in data.columns, f"Missing required column: {col}"

    train_data = data[data["is_train"] == 1]
    test_data = data[data["is_train"] == 0]

    assert train_data["Weekly_Sales"].isnull().sum() == 0, (
        "Training rows have missing Weekly_Sales."
    )

    assert test_data["Weekly_Sales"].isnull().sum() == len(test_data), (
        "Test rows should have missing Weekly_Sales because they are used for forecasting."
    )