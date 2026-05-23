def add_rul(df):
    max_cycle = (
        df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )
    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df
