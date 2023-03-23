import numpy as np
import pandas as pd


def to_submission_df(df) -> pd.DataFrame:
    if "azimuth" in df.columns and "zenith" in df.columns:
        # 天頂角、方位角の推論結果を 座標に変換する
        df["direction_x"] = np.cos(df["azimuth"]) * np.sin(df["zenith"])
        df["direction_y"] = np.sin(df["azimuth"]) * np.sin(df["zenith"])
        df["direction_z"] = np.cos(df["zenith"])

        # 座標推論結果と平均する
        # df["direction_x"] = (df["direction_x"] + df["direction_x_az"]) / 2
        # df["direction_y"] = (df["direction_y"] + df["direction_y_az"]) / 2
        # df["direction_z"] = (df["direction_z"] + df["direction_z_az"]) / 2

    else:
        # 座標を天頂角、方位角に変換する
        r = np.sqrt(df["direction_x"] ** 2 + df["direction_y"] ** 2 + df["direction_z"] ** 2)

        df.loc[:, "zenith"] = np.arccos(df["direction_z"] / r)
        df.loc[:, "azimuth"] = np.arctan2(df["direction_y"], df["direction_x"])

    # 方位角を 0-360 に補正
    df.loc[df["azimuth"] < 0, "azimuth"] = df["azimuth"][df["azimuth"] < 0] + 2 * np.pi

    drop_these_columns = []
    for column in df.columns:
        if column not in ["event_id", "zenith", "azimuth"]:
            drop_these_columns.append(column)

    return (
        df.drop(columns=drop_these_columns)
        .loc[:, ["event_id", "azimuth", "zenith"]]
        .sort_values("event_id")
        .set_index("event_id")
    )
