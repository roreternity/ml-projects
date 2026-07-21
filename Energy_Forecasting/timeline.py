from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "public_data.csv"
DELTA_WINDOW = 9  # окно в годах для усреднения дельты, подобрано на backtest

TIMELINE_FULL_PATH = BASE_DIR / "timeline_full.csv"
BACKTEST_PATH = BASE_DIR / "backtest_mean_delta_9_2017_2020.csv"
BASELINE_BACKTEST_PATH = BASE_DIR / "backtest_baseline_2017_2020.csv"
SMAPE_BY_STATE_PATH = BASE_DIR / "smape_by_state_2017_2020.csv"
FORECAST_PATH = BASE_DIR / "forecast_mean_delta_9_2020_2023.csv"


def smape(y_true, y_pred):
    """Симметричная процентная ошибка — одинаково штрафует пере- и недопрогноз."""
    denominator = y_true.abs() + y_pred.abs()
    result = (2 * (y_true - y_pred).abs() / denominator).where(denominator != 0, 0)
    return result.mean() * 100


def get_value_months_ago(state_history, date, months):
    """Значение ряда N месяцев назад. Если даты нет — берём ближайшее прошлое значение."""
    previous_date = date - pd.DateOffset(months=months)
    value = state_history.loc[state_history["date"] == previous_date, "value"]

    if len(value):
        return value.iloc[0]

    previous_values = state_history[state_history["date"] < date]["value"]
    return previous_values.iloc[-1] if len(previous_values) else 0


def get_month_delta_history(state_history, date):
    """История годовых дельт (год к году) для конкретного месяца."""
    month_history = state_history[
        (state_history["date"] < date)
        & (state_history["date"].dt.month == date.month)
    ].copy()

    month_history["lag_12"] = month_history["value"].shift(1)
    month_history["delta_12"] = month_history["value"] - month_history["lag_12"]
    return month_history.dropna()


def predict_delta_by_recent_mean(state_history, date, window=DELTA_WINDOW):
    """Средняя годовая дельта за последние `window` лет для этого месяца."""
    month_history = get_month_delta_history(state_history, date)
    recent_deltas = month_history["delta_12"].tail(window)

    if len(recent_deltas):
        return recent_deltas.mean()

    return 0


def mean_delta_forecast(history, states, future_dates, window=DELTA_WINDOW):
    """Прогноз mean_delta: сезонный baseline + поправка на среднюю дельту.
    Каждый следующий месяц прогнозируется рекурсивно — предсказание добавляется
    в историю и используется как основа для прогноза следующего шага.
    """
    predictions = []

    for date in future_dates:
        rows = []

        for state in states:
            state_history = history[history["state"] == state].sort_values("date")
            lag_12 = get_value_months_ago(state_history, date, 12)
            predicted_delta_12 = predict_delta_by_recent_mean(
                state_history,
                date,
                window=window,
            )
            prediction = lag_12 + predicted_delta_12

            rows.append({
                "date": date,
                "state": state,
                "prediction": prediction,
                "predicted_delta_12": predicted_delta_12,
                "lag_12": lag_12,
            })

        future_df = pd.DataFrame(rows)
        predictions.append(future_df)

        new_history = future_df[["date", "state", "prediction"]].rename(
            columns={"prediction": "value"}
        )
        history = pd.concat([history, new_history], ignore_index=True)

    return pd.concat(predictions, ignore_index=True)


def seasonal_baseline_forecast(history, states, future_dates):
    """Простой baseline для сравнения: прогноз = значение того же месяца год назад."""
    predictions = []

    for date in future_dates:
        rows = []

        for state in states:
            state_history = history[history["state"] == state].sort_values("date")
            prediction = get_value_months_ago(state_history, date, 12)

            rows.append({
                "date": date,
                "state": state,
                "baseline": prediction,
            })

        future_df = pd.DataFrame(rows)
        predictions.append(future_df)

        new_history = future_df[["date", "state", "baseline"]].rename(
            columns={"baseline": "value"}
        )
        history = pd.concat([history, new_history], ignore_index=True)

    return pd.concat(predictions, ignore_index=True)


# '--' и 'NM' — обозначения пропусков в исходном датасете
df = pd.read_csv(DATA_PATH, na_values=["--", "NM"])
df["date"] = pd.to_datetime(df["date"], format="%b %Y")

# Все колонки, кроме date, — значения временных рядов
value_columns = df.columns.drop("date")
df[value_columns] = df[value_columns].apply(pd.to_numeric, errors="coerce")

# wide -> long: одна строка = один ряд в один месяц. Удобно обрабатывать
# все 61 ряд одним циклом вместо 61 отдельной колонки
df_long = df.melt(
    id_vars="date",
    var_name="state",
    value_name="value",
)
df_long = df_long.sort_values(["state", "date"])
df_long["month"] = df_long["date"].dt.month
df_long["year"] = df_long["date"].dt.year
df_long["lag_12"] = df_long.groupby("state")["value"].shift(12)
df_long["delta_12"] = df_long["value"] - df_long["lag_12"]
df_long.to_csv(TIMELINE_FULL_PATH, index=False)

states = sorted(df_long["state"].unique())

# Backtest: обучаемся на данных до марта 2017, прогнозируем известный период
# (март 2017 — февраль 2020), сравниваем прогноз с реальными значениями
backtest_dates = pd.date_range("2017-03-01", "2020-02-01", freq="MS")
backtest_history = df_long[
    (df_long["date"] < "2017-03-01")
    & df_long["value"].notna()
][["date", "state", "value"]].copy()

backtest_forecast = mean_delta_forecast(
    backtest_history,
    states,
    backtest_dates,
    window=DELTA_WINDOW,
)
baseline_backtest = seasonal_baseline_forecast(
    backtest_history,
    states,
    backtest_dates,
)
baseline_backtest.to_csv(BASELINE_BACKTEST_PATH, index=False)

actual_backtest = df_long[
    (df_long["date"] >= "2017-03-01")
    & (df_long["date"] <= "2020-02-01")
][["date", "state", "value"]]

backtest_result = backtest_forecast.merge(
    actual_backtest,
    on=["date", "state"],
    how="inner",
).merge(
    baseline_backtest,
    on=["date", "state"],
    how="inner",
).dropna()
backtest_result.to_csv(BACKTEST_PATH, index=False)

model_mae = mean_absolute_error(backtest_result["value"], backtest_result["prediction"])
baseline_mae = mean_absolute_error(backtest_result["value"], backtest_result["baseline"])
model_rmse = mean_squared_error(backtest_result["value"], backtest_result["prediction"]) ** 0.5
baseline_rmse = mean_squared_error(backtest_result["value"], backtest_result["baseline"]) ** 0.5
model_smape = smape(backtest_result["value"], backtest_result["prediction"])
baseline_smape = smape(backtest_result["value"], backtest_result["baseline"])

# SMAPE по каждому ряду отдельно — показывает, где метод выигрывает у baseline, а где нет
smape_by_state = (
    backtest_result.groupby("state")
    .apply(
        lambda x: pd.Series({
            "mean_delta_9_smape": smape(x["value"], x["prediction"]),
            "baseline_smape": smape(x["value"], x["baseline"]),
        }),
        include_groups=False,
    )
    .reset_index()
)
smape_by_state["best_method"] = smape_by_state[
    ["mean_delta_9_smape", "baseline_smape"]
].idxmin(axis=1)
smape_by_state.to_csv(SMAPE_BY_STATE_PATH, index=False)

# Итоговый прогноз на реальное будущее — уже на всей доступной истории
future_dates = pd.date_range("2020-03-01", "2023-02-01", freq="MS")
history = df_long[["date", "state", "value"]].dropna().copy()
forecast = mean_delta_forecast(history, states, future_dates, window=DELTA_WINDOW)
forecast.to_csv(FORECAST_PATH, index=False)

print("Backtest 2017-03 — 2020-02")
print("Mean delta 9 MAE:", model_mae)
print("Baseline MAE:", baseline_mae)
print("Mean delta 9 RMSE:", model_rmse)
print("Baseline RMSE:", baseline_rmse)
print("Mean delta 9 SMAPE:", model_smape)
print("Baseline SMAPE:", baseline_smape)
print()
print("Форма итогового прогноза:", forecast.shape)
print("Прогноз сохранён в:", FORECAST_PATH)
