import numpy as np
import pandas as pd
import holidays


def update_df(old_df, new_df):
    old_df = old_df.copy()
    new_df = new_df.copy()

    old_df.index = pd.to_datetime(old_df.index)
    new_df.index = pd.to_datetime(new_df.index)

    old_df.update(new_df)

    is_new = ~new_df.index.isin(old_df.index)
    new_only = new_df[is_new]

    updated_df = pd.concat([old_df, new_only])

    updated_df.sort_index(inplace=True)
    updated_df.ffill(inplace=True)
    updated_df.bfill(inplace=True)

    return updated_df


def enrich_calendar(df_input):
    df = df_input.copy()
    df.index = pd.to_datetime(df.index)

    df["month_name"] = df.index.to_series().dt.strftime("%B")
    df["week_of_month"] = df.index.to_series().apply(week_of_month)
    df["day_of_week"] = df.index.to_series().dt.strftime("%A")
    df["is_weekend"] = df.index.to_series().dt.weekday.isin([5, 6]).astype(int)

    anos = df.index.year.unique()
    br_holidays = holidays.Brazil(years=anos)
    df["is_holiday"] = df.index.to_series().apply(lambda x: int(x in br_holidays))

    return df.loc[(df["is_weekend"].astype(bool) == False) & (df["is_holiday"].astype(bool) == False)]


def week_of_month(x):
    first_day = x.replace(day=1)
    dom = first_day.weekday()
    return int(np.ceil((x.day + dom) / 7.0))


def get_election_flags(year):
    is_general = 1 if (year - 2) % 4 == 0 else 0
    is_municipal = 1 if year % 4 == 0 else 0
    return is_general, is_municipal


def check_near_election(date):
    year = date.year
    is_general, is_municipal = get_election_flags(year)
    if not (is_general or is_municipal):
        return 0, 0
    first_turn = pd.date_range(start=f"{year}-10-01", end=f"{year}-10-07", freq="W-SUN")[0]
    second_turn = pd.date_range(start=f"{year}-10-25", end=f"{year}-10-31", freq="W-SUN")[0]
    near_first = int(abs((date - first_turn).days) <= 7)
    near_second = int(abs((date - second_turn).days) <= 7)
    return near_first, near_second

