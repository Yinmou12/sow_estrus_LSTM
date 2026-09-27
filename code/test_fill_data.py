"""Run with: python -B -m unittest -v test_fill_data."""

import unittest

import pandas as pd

from sow_estrus_LSTM_Function import fill_data


def sample(sow_no, split_id, start, base=38.0, positive=False):
    data = pd.DataFrame(
        {
            "sSowsNo": sow_no,
            "sSowsNo_split": split_id,
            "tLastUploadTime": pd.date_range(start, periods=48, freq="h"),
            "iTemperature": [base + hour / 100 for hour in range(48)],
            "iStep": [10.0] * 48,
            "temperatureRate": [0.01] * 48,
            "isEstrus": [0] * 47 + [int(positive)],
        }
    )
    return data


class FillDataTests(unittest.TestCase):
    def test_two_estrus_episodes_of_one_sow_are_filled_independently(self):
        first = sample("1121", "1121_1", "2024-11-10 16:00", positive=True).drop(index=12)
        second = sample("1121", "1121_2", "2025-04-10 16:00", 40.0, True).drop(index=20)
        negative = sample("9999", "9999_M_1", "2026-06-06 09:00")
        data = pd.concat([first, second, negative], ignore_index=True)
        result = fill_data(data)
        self.assertEqual(result.groupby("sSowsNo_split").size().to_dict(), {"1121_1": 48, "1121_2": 48, "9999_M_1": 48})
        self.assertEqual(len(result), 144)
        first_result = result.loc[result["sSowsNo_split"].eq("1121_1")].set_index("tLastUploadTime")
        second_result = result.loc[result["sSowsNo_split"].eq("1121_2")].set_index("tLastUploadTime")
        self.assertAlmostEqual(first_result.loc["2024-11-11 04:00", "iTemperature"], 38.11)
        self.assertAlmostEqual(second_result.loc["2025-04-11 12:00", "iTemperature"], 40.19)
        self.assertEqual(first_result["sSowsNo"].unique().tolist(), ["1121"])
        self.assertEqual(second_result["sSowsNo"].unique().tolist(), ["1121"])

    def test_overlapping_morning_and_afternoon_samples_keep_their_own_values(self):
        morning = sample("9999", "9999_M_1", "2026-06-06 09:00", 38.0).drop(index=12)
        afternoon = sample("9999", "9999_A_1", "2026-06-06 16:00", 40.0)
        afternoon["iStep"] = 100.0
        result = fill_data(pd.concat([morning, afternoon], ignore_index=True))
        self.assertEqual(result.groupby("sSowsNo_split").size().to_dict(), {"9999_A_1": 48, "9999_M_1": 48})
        common_time = pd.Timestamp("2026-06-06 21:00")
        at_time = result.loc[result["tLastUploadTime"].eq(common_time)].set_index("sSowsNo_split")
        self.assertAlmostEqual(at_time.loc["9999_M_1", "iTemperature"], 38.11)
        self.assertAlmostEqual(at_time.loc["9999_A_1", "iTemperature"], 40.05)
        self.assertEqual(at_time.loc["9999_M_1", "iStep"], 10.0)
        self.assertEqual(at_time.loc["9999_A_1", "iStep"], 100.0)
        self.assertEqual(at_time.loc["9999_M_1", "temperatureRate"], 0.0)
        self.assertEqual(at_time.loc["9999_M_1", "isEstrus"], 0)

    def test_positive_only_input_and_inserted_identifiers_are_supported(self):
        data = sample("1121", "1121_1", "2024-11-10 16:00", positive=True).drop(index=12)
        result = fill_data(data)
        self.assertEqual(len(result), 48)
        self.assertFalse(result[["sSowsNo", "sSowsNo_split"]].isna().any().any())
        self.assertEqual(result["sSowsNo_split"].unique().tolist(), ["1121_1"])
        self.assertEqual(result["isEstrus"].sum(), 1)

    def test_old_balancing_and_stride_arguments_do_not_change_selected_samples(self):
        positive = sample("1121", "1121_1", "2024-11-10 16:00", positive=True)
        negative = sample("9999", "9999_M_1", "2026-06-06 09:00")
        data = pd.concat([positive, negative], ignore_index=True)
        default = fill_data(data)
        alternate = fill_data(data, balanced_data=False, stride=1)
        pd.testing.assert_frame_equal(default, alternate)
        self.assertEqual(set(default["sSowsNo_split"]), {"1121_1", "9999_M_1"})
        self.assertEqual(set(default["sSowsNo"]), {"1121", "9999"})

    def test_input_is_not_mutated_and_existing_observations_are_preserved(self):
        data = sample("9999", "9999_M_1", "2026-06-06 09:00").drop(index=12)
        data = data.sample(frac=1, random_state=7)
        original = data.copy(deep=True)
        result = fill_data(data)
        pd.testing.assert_frame_equal(data, original)
        ordered = original.sort_values("tLastUploadTime").set_index("tLastUploadTime")
        existing = result.set_index("tLastUploadTime").loc[ordered.index, ordered.columns]
        pd.testing.assert_frame_equal(existing, ordered, check_dtype=False)

    def test_missing_first_and_last_hours_are_filled_to_the_morning_boundary(self):
        data = sample("9999", "9999_M_1", "2026-06-06 09:00").drop(index=[0, 1, 46, 47])
        result = fill_data(data).sort_values("tLastUploadTime")
        self.assertEqual(len(result), 48)
        self.assertEqual(result["tLastUploadTime"].iloc[0], pd.Timestamp("2026-06-06 09:00"))
        self.assertEqual(result["tLastUploadTime"].iloc[-1], pd.Timestamp("2026-06-08 08:00"))
        self.assertAlmostEqual(result["iTemperature"].iloc[0], 38.02)
        self.assertAlmostEqual(result["iTemperature"].iloc[-1], 38.45)
        self.assertFalse(result[["sSowsNo", "sSowsNo_split", "iTemperature"]].isna().any().any())

    def test_estrus_label_sets_the_end_and_excludes_extra_future_record(self):
        data = sample("1121", "1121_1", "2024-11-10 16:00", positive=True)
        extra = data.iloc[-1:].copy()
        extra["tLastUploadTime"] += pd.Timedelta(hours=1)
        extra["iTemperature"] = 99.0
        extra["isEstrus"] = 0
        result = fill_data(pd.concat([data, extra], ignore_index=True))
        self.assertEqual(len(result), 48)
        self.assertEqual(result["tLastUploadTime"].max(), pd.Timestamp("2024-11-12 15:00"))
        self.assertLess(result["iTemperature"].max(), 40)

    def test_missing_estrus_endpoint_uses_the_unique_08_or_15_window(self):
        first = sample("1121", "1121_1", "2024-11-10 16:00", positive=True)
        second = sample("1121", "1121_2", "2025-04-10 16:00", positive=True).iloc[:-1]
        result = fill_data(pd.concat([first, second], ignore_index=True))
        second_result = result.loc[result["sSowsNo_split"].eq("1121_2")]
        self.assertEqual(len(second_result), 48)
        self.assertEqual(second_result["tLastUploadTime"].max(), pd.Timestamp("2025-04-12 15:00"))

    def test_short_and_constant_samples_are_removed_for_both_classes(self):
        positive = sample("1121", "1121_1", "2024-11-10 16:00", positive=True)
        negative = sample("9999", "9999_M_1", "2026-06-06 09:00")
        short_positive = positive.drop(index=[1, 2, 3, 4, 5])
        short_positive["sSowsNo_split"] = "1121_2"
        short_negative = negative.drop(index=[1, 2, 3, 4, 5])
        short_negative["sSowsNo_split"] = "9999_M_2"
        constant_positive = positive.copy()
        constant_positive["sSowsNo_split"] = "1121_3"
        constant_positive["iTemperature"] = 38.0
        constant_negative = negative.copy()
        constant_negative["sSowsNo_split"] = "9999_M_3"
        constant_negative["iTemperature"] = 38.0
        result = fill_data(pd.concat([positive,negative,short_positive,short_negative,constant_positive,constant_negative], ignore_index=True))
        self.assertEqual(set(result["sSowsNo_split"]), {"1121_1", "9999_M_1"})
        self.assertEqual(len(result), 96)

    def test_empty_and_all_discarded_inputs_keep_schema(self):
        data = sample("9999", "9999_M_1", "2026-06-06 09:00")
        for source in (data.iloc[:0], data.iloc[:43]):
            result = fill_data(source)
            self.assertTrue(result.empty)
            self.assertEqual(result.columns.tolist(), data.columns.tolist())


if __name__ == "__main__":
    unittest.main()
