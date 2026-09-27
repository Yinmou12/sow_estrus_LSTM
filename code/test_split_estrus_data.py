"""Run with: python -B -m unittest -v test_split_estrus_data."""

import unittest

import pandas as pd

from sow_estrus_LSTM_Function import split_estrusData
from sow_estrus_LSTM_Info import DEL_CODE


def episode(sow_no, start, end, positive=True):
    times = pd.date_range(start, end, freq="h")
    data = pd.DataFrame(
        {
            "sSowsNo": sow_no,
            "tLastUploadTime": times,
            "iTemperature": [38 + i / 1000 for i in range(len(times))],
            "temperatureRate": 0.01,
            "isEstrus": 0,
            "sSowsNo_split": str(sow_no),
        }
    )
    if positive:
        data.loc[data.index[-1], "isEstrus"] = 1
    return data


def repeat_records(sow_no):
    return [f"2026-06-08_M_{sow_no}", f"2026-07-08_A_{sow_no}"]


class SplitEstrusDataTests(unittest.TestCase):
    def test_three_episodes_change_split_ids_only_in_chronological_order(self):
        parts = [
            episode("9999", "2026-06-06 09:00", "2026-06-08 08:00"),
            episode("9999", "2026-07-06 16:00", "2026-07-08 15:00"),
            episode("9999", "2026-08-06 09:00", "2026-08-08 08:00"),
        ]
        data = pd.concat(parts).sample(frac=1, random_state=7)
        result = split_estrusData(data, repeat_records("9999") + ["2026-08-08_M_9999"], 7)
        expected = pd.concat(parts, ignore_index=True)
        expected["sSowsNo_split"] = ["9999_1"] * 48 + ["9999_2"] * 48 + ["9999_3"] * 48
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_missing_hours_and_missing_endpoint_label_are_retained(self):
        first = episode("9999", "2026-06-06 09:00", "2026-06-08 08:00")
        first = first.drop(index=[10, 20, 47])
        second = episode("9999", "2026-07-06 16:00", "2026-07-08 15:00")
        data = pd.concat([first, second], ignore_index=True)
        result = split_estrusData(data, repeat_records("9999"), 7)
        self.assertEqual(result.groupby("sSowsNo_split").size().to_dict(), {"9999_1": 45, "9999_2": 48})
        pd.testing.assert_frame_equal(result.drop(columns="sSowsNo_split").reset_index(drop=True), data.drop(columns="sSowsNo_split"))

    def test_negative_morning_afternoon_windows_are_not_overwritten(self):
        morning = episode("8888", "2026-06-06 09:00", "2026-06-08 08:00", False)
        afternoon = episode("8888", "2026-06-06 16:00", "2026-06-08 15:00", False)
        morning["sSowsNo_split"] = "8888_M_1"
        afternoon["sSowsNo_split"] = "8888_A_1"
        data = pd.concat([morning, afternoon], ignore_index=True)
        # Even a repeated configuration entry must not relabel selected negatives.
        result = split_estrusData(data, repeat_records("8888"), 7)
        keys = ["sSowsNo_split", "tLastUploadTime"]
        pd.testing.assert_frame_equal(result.sort_values(keys).reset_index(drop=True), data.sort_values(keys).reset_index(drop=True))

    def test_one_available_episode_keeps_original_split_id(self):
        data = episode("9999", "2026-06-06 09:00", "2026-06-08 08:00")
        result = split_estrusData(data, repeat_records("9999"), 7)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), data)

    def test_exact_threshold_does_not_split_but_larger_gap_does(self):
        data = episode("9999", "2026-06-01 08:00", "2026-06-01 08:00")
        data = pd.concat([data] * 3, ignore_index=True)
        data["tLastUploadTime"] = pd.to_datetime(["2026-06-01 08:00", "2026-06-08 08:00", "2026-06-15 09:00"])
        result = split_estrusData(data, repeat_records("9999"), 7)
        self.assertEqual(result["sSowsNo_split"].tolist(), ["9999_1", "9999_1", "9999_2"])

    def test_input_is_not_mutated_and_repeated_calls_do_not_append_suffixes(self):
        data = pd.concat([
            episode("9999", "2026-06-06 09:00", "2026-06-08 08:00"),
            episode("9999", "2026-07-06 16:00", "2026-07-08 15:00"),
        ], ignore_index=True)
        data["sSowsNo"] = 9999
        data["tLastUploadTime"] = data["tLastUploadTime"].astype(str)
        original = data.copy(deep=True)
        result = split_estrusData(data, repeat_records("9999"), 7)
        pd.testing.assert_frame_equal(data, original)
        repeated = split_estrusData(result, repeat_records("9999"), 7)
        pd.testing.assert_frame_equal(repeated, result)

    def test_no_repeated_configuration_and_empty_input_return_normally(self):
        data = episode("9999", "2026-06-06 09:00", "2026-06-08 08:00")
        pd.testing.assert_frame_equal(split_estrusData(data, ["2026-06-08_M_9999"], 7), data)
        empty = data.iloc[:0]
        pd.testing.assert_frame_equal(split_estrusData(empty, repeat_records("9999"), 7), empty)

    def test_missing_split_column_is_initialized_from_original_sow_id(self):
        data = episode("9999", "2026-06-06 09:00", "2026-06-08 08:00").drop(columns="sSowsNo_split")
        result = split_estrusData(data, [], 7)
        self.assertEqual(result["sSowsNo_split"].unique().tolist(), ["9999"])
        pd.testing.assert_frame_equal(result[data.columns], data)

    def test_excluded_sow_keeps_existing_records(self):
        self.assertTrue(DEL_CODE)
        sow_no = str(DEL_CODE[0])
        data = pd.concat([
            episode(sow_no, "2026-06-06 09:00", "2026-06-08 08:00"),
            episode(sow_no, "2026-07-06 16:00", "2026-07-08 15:00"),
        ], ignore_index=True)
        result = split_estrusData(data, repeat_records(sow_no), 7)
        pd.testing.assert_frame_equal(result, data)


if __name__ == "__main__":
    unittest.main()
