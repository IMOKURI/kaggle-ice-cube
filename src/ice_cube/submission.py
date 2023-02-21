import numpy as np
import pandas as pd


def to_submission_df(df, angle_post_fix="", vec_post_fix="") -> pd.DataFrame:
    r = np.sqrt(
        df["direction_x" + vec_post_fix] ** 2
        + df["direction_y" + vec_post_fix] ** 2
        + df["direction_z" + vec_post_fix] ** 2
    )
    df["zenith" + angle_post_fix] = np.arccos(df["direction_z" + vec_post_fix] / r)
    df["azimuth" + angle_post_fix] = np.arctan2(
        df["direction_y" + vec_post_fix], df["direction_x" + vec_post_fix]
    )  # np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    df["azimuth" + angle_post_fix][df["azimuth" + angle_post_fix] < 0] = (
        df["azimuth" + angle_post_fix][df["azimuth" + angle_post_fix] < 0] + 2 * np.pi
    )

    drop_these_columns = []
    for column in df.columns:
        if column not in ["event_id", "zenith", "azimuth"]:
            drop_these_columns.append(column)
    return df.drop(columns=drop_these_columns).iloc[:, [0, 2, 1]].set_index("event_id")
