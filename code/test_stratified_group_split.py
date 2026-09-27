"""Run with: python -B -m unittest -v test_stratified_group_split."""

import contextlib
import io
import re
import unittest

import pandas as pd

from sow_estrus_LSTM_Function import stratified_group_split


def make_dataset():
    rows = []
    for index in range(10):
        for positive, prefix, suffixes in (
            (True, "P", ("1", "2")),
            (False, "N", ("M_1", "A_1", "M_2")),
        ):
            sow_no = f"{prefix}{index:02d}"
            for suffix in suffixes:
                for hour in range(3):
                    rows.append(
                        {
                            "sSowsNo": sow_no,
                            "sSowsNo_split": f"{sow_no}_{suffix}",
                            "tLastUploadTime": pd.Timestamp("2026-06-01") + pd.Timedelta(hours=hour),
                            # The second estrus episode has no positive endpoint label.
                            "isEstrus": int(positive and suffix == "1" and hour == 2),
                            "iTemperature": 38 + hour / 10,
                            "row_id": len(rows),
                        }
                    )
    return pd.DataFrame(rows)


class StratifiedGroupSplitTests(unittest.TestCase):
    def test_printed_counts_use_samples_and_inherit_missing_endpoint_class(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stratified_group_split(make_dataset(), 0.6, 0.2, 0.2)
        expected = {"训练集": [12, 30, 12, 18], "验证集": [4, 10, 4, 6], "测试集": [4, 10, 4, 6]}
        lines = output.getvalue().splitlines()
        for name, counts in expected.items():
            line = next(line for line in lines if line.startswith(name))
            self.assertEqual([int(number) for number in re.findall(r"\d+", line)], counts)

    def test_all_rows_and_sow_samples_stay_in_exactly_one_partition(self):
        data = make_dataset()
        # Duplicate DataFrame indices must not be treated as duplicate observations.
        data.index = [index % 7 for index in range(len(data))]
        original = data.copy(deep=True)
        parts = stratified_group_split(data, 0.6, 0.2, 0.2)
        for i, first in enumerate(parts):
            for second in parts[i + 1:]:
                self.assertTrue(set(first["sSowsNo"]).isdisjoint(second["sSowsNo"]))
                self.assertTrue(set(first["sSowsNo_split"]).isdisjoint(second["sSowsNo_split"]))
        restored = pd.concat(parts).sort_values("row_id")
        pd.testing.assert_frame_equal(restored, original)
        pd.testing.assert_frame_equal(data, original)

    def test_same_input_has_reproducible_assignment(self):
        data = make_dataset()
        first = stratified_group_split(data)
        second = stratified_group_split(data)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)

    def test_rejects_a_sample_id_shared_by_different_sows(self):
        data = make_dataset()
        data.loc[data["sSowsNo_split"].isin(["P00_1", "P01_1"]), "sSowsNo_split"] = "shared"
        with self.assertRaisesRegex(ValueError, "sSowsNo_split"):
            stratified_group_split(data)

    def test_rejects_missing_identifier_columns_and_values(self):
        for column in ("sSowsNo", "sSowsNo_split"):
            with self.subTest(column=column, case="missing column"):
                with self.assertRaisesRegex(ValueError, column):
                    stratified_group_split(make_dataset().drop(columns=column))
            with self.subTest(column=column, case="missing value"):
                data = make_dataset()
                data.loc[0, column] = None
                with self.assertRaisesRegex(ValueError, column):
                    stratified_group_split(data)

    def test_rejects_ratios_that_do_not_form_three_positive_parts(self):
        for ratios in ((0.7, 0.2, 0.2), (0.7, 0, 0.3), (0.7, -0.1, 0.4), (float("nan"), 0.1, 0.2)):
            with self.subTest(ratios=ratios):
                with self.assertRaisesRegex(ValueError, "比例"):
                    stratified_group_split(make_dataset(), *ratios)


if __name__ == "__main__":
    unittest.main()
