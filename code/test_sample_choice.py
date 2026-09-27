"""Run with: python -B -m unittest -v test_sample_choice."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data_preparation
from data_preparation import sample_choice


def hourly_data(start, end, sow_no="9999"):
    times = pd.date_range(start, end, freq="h")
    return pd.DataFrame(
        {
            "sSowsNo": sow_no,
            "tLastUploadTime": times,
            "iTemperature": [38 + i / 1000 for i in range(len(times))],
            "temperatureRate": 0.01,
            "isEstrus": 0,
        }
    )


class SampleChoiceTests(unittest.TestCase):
    def select(self, data):
        return sample_choice(data)

    def test_example_boundaries_and_separate_morning_afternoon_sequences(self):
        data = hourly_data("2026-06-06 00:00", "2026-06-16 00:00")
        result = self.select(data.sample(frac=1, random_state=7))
        groups = result.groupby("sSowsNo_split")
        expected = {
            "9999_M_1": ("2026-06-06 09:00", "2026-06-08 08:00"),
            "9999_M_2": ("2026-06-08 09:00", "2026-06-10 08:00"),
            "9999_M_3": ("2026-06-10 09:00", "2026-06-12 08:00"),
            "9999_M_4": ("2026-06-12 09:00", "2026-06-14 08:00"),
            "9999_A_1": ("2026-06-06 16:00", "2026-06-08 15:00"),
            "9999_A_2": ("2026-06-08 16:00", "2026-06-10 15:00"),
            "9999_A_3": ("2026-06-10 16:00", "2026-06-12 15:00"),
            "9999_A_4": ("2026-06-12 16:00", "2026-06-14 15:00"),
        }
        self.assertEqual(set(groups.groups), set(expected))
        for split_id, (start, end) in expected.items():
            group = groups.get_group(split_id)
            self.assertEqual(len(group), 48)
            self.assertEqual(group["tLastUploadTime"].min(), pd.Timestamp(start))
            self.assertEqual(group["tLastUploadTime"].max(), pd.Timestamp(end))
            self.assertEqual(group["sSowsNo"].unique().tolist(), ["9999"])
        for period in ("M", "A"):
            part = result[result["sSowsNo_split"].str.contains(f"_{period}_")]
            self.assertFalse(part["tLastUploadTime"].duplicated().any())
        self.assertTrue(result["tLastUploadTime"].duplicated().any())

    def test_missing_hours_are_retained_without_filling_or_shifting_windows(self):
        data = hourly_data("2026-06-06 00:00", "2026-06-10 08:00")
        missing = pd.Timestamp("2026-06-07 12:00")
        data = data[data["tLastUploadTime"] != missing]
        result = self.select(data)
        first = result[result["sSowsNo_split"] == "9999_M_1"]
        second = result[result["sSowsNo_split"] == "9999_M_2"]
        self.assertEqual(len(first), 47)
        self.assertNotIn(missing, set(first["tLastUploadTime"]))
        self.assertEqual(len(second), 48)
        self.assertEqual(second["tLastUploadTime"].min(), pd.Timestamp("2026-06-08 09:00"))

    def test_last_complete_morning_is_kept_and_incomplete_afternoon_is_dropped(self):
        data = hourly_data("2026-06-06 09:00", "2026-06-08 08:00")
        result = self.select(data)
        self.assertEqual(result["sSowsNo_split"].unique().tolist(), ["9999_M_1"])
        self.assertEqual(len(result), 48)
        self.assertTrue(self.select(data.iloc[:-1]).empty)

    def test_late_start_uses_next_eligible_daily_anchor(self):
        data = hourly_data("2026-06-06 10:00", "2026-06-09 15:00")
        result = self.select(data)
        morning = result[result["sSowsNo_split"] == "9999_M_1"]
        afternoon = result[result["sSowsNo_split"] == "9999_A_1"]
        self.assertEqual(morning["tLastUploadTime"].min(), pd.Timestamp("2026-06-07 09:00"))
        self.assertEqual(morning["tLastUploadTime"].max(), pd.Timestamp("2026-06-09 08:00"))
        self.assertEqual(afternoon["tLastUploadTime"].min(), pd.Timestamp("2026-06-06 16:00"))
        self.assertEqual(afternoon["tLastUploadTime"].max(), pd.Timestamp("2026-06-08 15:00"))

    def test_positive_sow_keeps_all_rows_and_features_and_input_is_not_mutated(self):
        positive = hourly_data("2026-06-06 00:00", "2026-06-10 08:00", "0017")
        positive.loc[positive.index[-1], "isEstrus"] = 1
        negative = hourly_data("2026-06-06 00:00", "2026-06-10 08:00")
        data = pd.concat([negative, positive], ignore_index=True)
        original = data.copy(deep=True)
        result = self.select(data)
        actual = result[result["sSowsNo"] == "0017"].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[positive.columns], positive)
        self.assertEqual(actual["sSowsNo_split"].unique().tolist(), ["0017"])
        pd.testing.assert_frame_equal(data, original)

    def test_numbering_restarts_per_sow_and_empty_windows_are_skipped(self):
        sparse = pd.concat(
            [hourly_data("2026-06-06 09:00", "2026-06-08 08:00"),
             hourly_data("2026-06-12 09:00", "2026-06-14 08:00")],
            ignore_index=True,
        )
        other = hourly_data("2026-06-06 09:00", "2026-06-08 08:00", "1234")
        result = self.select(pd.concat([sparse, other], ignore_index=True))
        morning_ids = result.loc[result["sSowsNo_split"].str.contains("_M_"), "sSowsNo_split"]
        self.assertEqual(set(morning_ids), {"9999_M_1", "9999_M_2", "1234_M_1"})
        later = result[result["sSowsNo_split"] == "9999_M_2"]
        self.assertEqual(later["tLastUploadTime"].min(), pd.Timestamp("2026-06-12 09:00"))

    def test_empty_input_and_no_valid_negative_windows_keep_output_schema(self):
        data = hourly_data("2026-06-06 00:00", "2026-06-06 23:00")
        for source in (data, data.iloc[:0]):
            result = self.select(source)
            self.assertTrue(result.empty)
            self.assertEqual(result.columns.tolist(), [*data.columns, "sSowsNo_split"])

    def test_default_reads_requested_workbook_and_keeps_text_sow_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "processed_dataset"
            source_dir.mkdir()
            data = hourly_data("2026-06-06 09:00", "2026-06-08 08:00", "0017")
            data.to_excel(source_dir / "processed_dataset_2026_0913_1609.xlsx", index=False)
            with patch.object(data_preparation, "experimentRecord_data_path", directory):
                result = sample_choice()
            self.assertEqual(result["sSowsNo_split"].unique().tolist(), ["0017_M_1"])
            self.assertEqual(result["sSowsNo"].unique().tolist(), ["0017"])
            self.assertEqual(len(result), 48)


if __name__ == "__main__":
    unittest.main()
