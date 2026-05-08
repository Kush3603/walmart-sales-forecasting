# This module contains functions to build features for the Walmart Sales Forecasting project.
def add_time_features(df):
    df = df.copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day

    return df # Add time-based features such as year, month, week, and day from the date column.

# Add lag features for the target variable (Weekly_Sales) to capture temporal dependencies. For example, you can create lag features for the previous week, two weeks ago, etc.
def add_lag_features(df):
    df = df.copy()

    df = df.sort_values(by=["Store", "Dept", "Date"])

    df["Lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
    df["Lag_2"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(2)
    df["Lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)

    return df

# Add rolling window features to capture trends and seasonality in the sales data. For example, you can create rolling mean and rolling standard deviation features over a 4-week window.
def add_rolling_features(df):
    df = df.copy()

    df = df.sort_values(by=["Store", "Dept", "Date"])

    df["Rolling_Mean_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(lambda x: x.shift(1).rolling(4).mean())
    df["Rolling_Std_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(lambda x: x.shift(1).rolling(4).std())

    return df

# Add features related to markdowns, such as the total markdown amount and the interaction between markdowns and holidays.
def add_markdown_features(df):
    df = df.copy()

    df["Total_MarkDown"] = df[
        ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    ].sum(axis=1)

    return df

# Add interaction features to capture the combined effect of different variables. For example, you can create an interaction feature between holidays and markdowns to see if markdowns have a different impact on sales during holiday weeks.
def add_interaction_features(df):
    df = df.copy()

    df["Holiday_Markdown"] = df["IsHoliday"] * df["Total_MarkDown"]

    return df

# Handle missing values in the lag and rolling features by filling them with 0, which indicates no previous sales data available for those periods.
def handle_missing_features(df):
    df = df.copy()

    lag_cols = [
        "Lag_1",
        "Lag_2",
        "Lag_4",
        "Rolling_Mean_4",
        "Rolling_Std_4"
    ]

    df[lag_cols] = df[lag_cols].fillna(0)

    return df

# Main function to build all features by calling the individual feature engineering functions in sequence.
def build_features(df):
    df = df.copy()

    df = add_time_features(df)
    df = add_markdown_features(df)
    df = add_interaction_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = handle_missing_features(df)

    return df