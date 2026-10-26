import pandas as pd

def build_sequence(df, seuil=60):

    df = df.sort_values("exif_datetime").copy()

    df["delta_sec"] = df["exif_datetime"].diff().dt.total_seconds().fillna(0)

    df["sequence_id"] = (df["delta_sec"] > seuil).cumsum()

    return df