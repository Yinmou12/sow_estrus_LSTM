"""Run with: python -B -m unittest -v test_processing_feature."""

import unittest
from unittest.mock import patch

import pandas as pd

import data_preparation as preparation


def sample(sow_no="9999", split_id="9999_M_1", start="2026-06-06 09:00", base=38.0):
    return pd.DataFrame({
        "sEarTagCode": "00123",
        "sSowsNo": sow_no,
        "sSowsNo_split": split_id,
        "sBrand": "A",
        "tLastUploadTime": pd.date_range(start, periods=48, freq="h"),
        "iTemperature": [base + hour / 100 for hour in range(48)],
        "temperatureRate": [hour / 1000 for hour in range(48)],
        "isEstrus": [0] * 48,
        "iStep": [10] * 48,
    })


class ProcessingFeatureTests(unittest.TestCase):
    def test_default_file_uses_original_sow_class_when_endpoint_label_was_lost(self):
        data = sample("1121", "1121_2")
        labels = pd.DataFrame({"sSowsNo": ["1121"], "isEstrus": [1]})
        # 固定文件输入；排序、展开和样本标签计算均执行真实函数。
        with patch.object(preparation.pd, "read_excel", side_effect=[data, labels]):
            result = preparation.processing_feature()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, 102))
        self.assertEqual(result.iloc[0]["isEstrus"], 1)
        self.assertEqual(result.iloc[0]["sEarTagCode"], "00123")

    def test_columns_values_and_end_time_follow_chronological_order(self):
        data = sample().sample(frac=1, random_state=7)
        original = data.copy(deep=True)
        result = preparation.processing_feature(data)
        self.assertEqual(result.shape, (1, 102))
        self.assertEqual(result.columns[:5].tolist(), [
            "sEarTagCode", "sSowsNo", "sSowsNo_split", "sBrand", "tLastUploadTime"
        ])
        self.assertEqual(result.columns[5:53].tolist(), [f"iTemperature_{i}" for i in range(1, 49)])
        self.assertEqual(result.columns[53:101].tolist(), [f"temperatureRate_{i}" for i in range(1, 49)])
        self.assertEqual(result.columns[-1], "isEstrus")
        self.assertEqual(result.iloc[0]["tLastUploadTime"], pd.Timestamp("2026-06-08 08:00"))
        self.assertEqual(result.iloc[0, 5:53].tolist(), [38.0 + hour / 100 for hour in range(48)])
        self.assertEqual(result.iloc[0, 53:101].tolist(), [hour / 1000 for hour in range(48)])
        self.assertEqual(result.iloc[0]["isEstrus"], 0)
        pd.testing.assert_frame_equal(data, original)

    def test_overlapping_windows_of_one_sow_remain_separate_samples(self):
        morning = sample()
        afternoon = sample(split_id="9999_A_1", start="2026-06-06 16:00", base=40.0)
        result = preparation.processing_feature(pd.concat([morning, afternoon], ignore_index=True))
        self.assertEqual(result["sSowsNo"].tolist(), ["9999", "9999"])
        self.assertEqual(result["sSowsNo_split"].tolist(), ["9999_M_1", "9999_A_1"])
        self.assertEqual(result["iTemperature_1"].tolist(), [38.0, 40.0])
        self.assertEqual(result["tLastUploadTime"].dt.hour.tolist(), [8, 15])

    def test_multiple_estrus_episodes_use_the_established_sow_class(self):
        first = sample("1121", "1121_1")
        first.loc[47, "isEstrus"] = 1
        second = sample("1121", "1121_2", "2026-07-06 09:00", 40.0)
        negative = sample()
        result = preparation.processing_feature(pd.concat([first, second, negative], ignore_index=True))
        self.assertEqual(result["isEstrus"].tolist(), [1, 1, 0])
        self.assertEqual(result["sSowsNo_split"].tolist(), ["1121_1", "1121_2", "9999_M_1"])

    def test_explicit_sow_classes_survive_removal_of_all_positive_observations(self):
        data = sample("1121", "1121_2")
        result = preparation.processing_feature(data, estrus_sows={"1121"})
        self.assertEqual(result.iloc[0]["isEstrus"], 1)

    def test_invalid_sample_lengths_times_and_identifiers_name_the_sample(self):
        invalid = {}
        invalid["short"] = sample().iloc[:-1]
        invalid["long"] = pd.concat([sample(), sample().iloc[:1]], ignore_index=True)
        invalid["duplicate"] = sample()
        invalid["duplicate"].loc[47, "tLastUploadTime"] = invalid["duplicate"].loc[46, "tLastUploadTime"]
        invalid["gap"] = sample()
        invalid["gap"].loc[47, "tLastUploadTime"] += pd.Timedelta(hours=1)
        invalid["mixed_sows"] = sample()
        invalid["mixed_sows"].loc[47, "sSowsNo"] = "8888"
        for name, data in invalid.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "9999_M_1"):
                    preparation.processing_feature(data)

    def test_missing_features_raise_instead_of_producing_incomplete_vectors(self):
        data = sample()
        data.loc[12, "temperatureRate"] = float("nan")
        with self.assertRaisesRegex(ValueError, "9999_M_1"):
            preparation.processing_feature(data)

    def test_empty_input_has_full_output_schema_and_missing_brand_is_preserved(self):
        data = sample()
        result = preparation.processing_feature(data.iloc[:0])
        self.assertEqual(result.shape, (0, 102))
        self.assertEqual(result.columns[-1], "isEstrus")
        data["sBrand"] = None
        result = preparation.processing_feature(data)
        self.assertTrue(pd.isna(result.iloc[0]["sBrand"]))


if __name__ == "__main__":
    unittest.main()
