"""验证新实验协议的窗口、分组、特征和预测导出约束。"""

import importlib.util
import contextlib
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import sow_estrus_LSTM_Function as legacy
from sow_estrus_LSTM_train import find_best_threshold


@contextlib.contextmanager
def test_directory():
    """在当前测试工作区建立普通权限目录，清理前验证绝对路径边界。"""
    parent = Path(__file__).resolve().parent
    directory = parent / ('test_workspace_' + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        resolved = directory.resolve()
        if not resolved.is_relative_to(parent) or resolved == parent:
            raise RuntimeError('测试清理路径超出工作区')
        shutil.rmtree(resolved)


def make_windows(sows=10):
    """生成每头母猪含两个重叠窗口的可重复测试数据，保留前导零编号。"""
    rows = []
    for sow in range(sows):
        for episode in range(2):
            for hour in range(48):
                rows.append({
                    'sEarTagCode': f'00{sow:03}', 'sSowsNo': f'{sow:04}',
                    'sSowsNo_split': f'{sow:04}_M_{episode+1}', 'sBrand': 'A',
                    'tLastUploadTime': pd.Timestamp('2026-06-01') + pd.Timedelta(hours=hour+episode*7),
                    'iTemperature': 37+sow*.01+episode*.3+hour*.02,
                    'temperatureRate': 999.0, 'iStep': 123,
                    'isEstrus': int(sow % 2 == 1 and hour == 47),
                })
    return pd.DataFrame(rows)


class ExistingIntegrationTests(unittest.TestCase):
    """旧公共转换接口必须保留同一母猪的每个独立窗口。"""

    def test_convert_preserves_all_windows(self):
        """旧训练转换不能把多窗口压缩成每头母猪一个样本。"""
        result = legacy.convert_features(make_windows())
        self.assertEqual(len(result), 20)

    def test_both_legacy_tensor_modes_preserve_windows_and_rates(self):
        """单、双通道转换都保留所有窗口并重算差分。"""
        data = make_windows().sample(frac=1, random_state=7)
        one, y1, _ = legacy.prepare_univariate_lstm_data(data)
        two, y2, scaler = legacy.prepare_lstm_data(data)
        self.assertEqual(one.shape, (20, 48, 1))
        self.assertEqual(two.shape, (20, 48, 2))
        np.testing.assert_array_equal(y1, y2)
        raw = scaler.inverse_transform(two.reshape(-1, 2)).reshape(20, 48, 2)
        np.testing.assert_allclose(raw[:, 0, 1], 0, atol=1e-5)
        np.testing.assert_allclose(raw[:, 1:, 1], np.diff(raw[:, :, 0], axis=1), atol=1e-5)

    def test_threshold_accepts_probabilities_and_resolves_ties(self):
        """概率输入不应触发分类指标类型异常，平局优先靠近0.5。"""
        self.assertAlmostEqual(find_best_threshold([0, 1], [.1, .9]), .5)


class ProtocolTests(unittest.TestCase):
    """统一窗口表示、组划分及可追溯输出的行为测试。"""

    def setUp(self):
        """确认协议模块已经提供，再载入接口。"""
        self.assertIsNotNone(importlib.util.find_spec('experiment_data'), '缺少统一实验数据模块')
        import experiment_data
        self.api = experiment_data

    def test_window_order_and_metadata_match(self):
        """输入打乱后仍按样本键排列，温度及编号不会错位。"""
        batch = self.api.build_windows(make_windows().sample(frac=1, random_state=11))
        self.assertEqual(batch.temperatures.shape, (20, 48))
        self.assertEqual(batch.metadata.iloc[0]['sSowsNo'], '0000')
        self.assertEqual(batch.metadata.iloc[0]['sEarTagCode'], '00000')
        np.testing.assert_allclose(batch.temperatures[1, 0], 37.3)
        self.assertEqual(batch.labels.sum(), 10)

    def test_invalid_window_cannot_silently_disappear(self):
        """不足48小时或重复小时必须带编号报错。"""
        data = make_windows()
        with self.assertRaisesRegex(ValueError, '0000_M_1'):
            self.api.build_windows(data.drop(index=0))
        data.loc[0, 'tLastUploadTime'] = data.loc[1, 'tLastUploadTime']
        with self.assertRaisesRegex(ValueError, '0000_M_1'):
            self.api.build_windows(data)

    def test_group_folds_cover_all_samples_exactly_once(self):
        """所有周期均随母猪归组，每个窗口恰好一次作为验证样本。"""
        batch = self.api.build_windows(make_windows())
        folds = self.api.group_folds(batch, n_splits=5, seed=54)
        seen = []
        for train, val in folds:
            self.assertTrue(set(batch.metadata.iloc[train].sSowsNo).isdisjoint(batch.metadata.iloc[val].sSowsNo))
            seen.extend(val)
        self.assertEqual(sorted(seen), list(range(20)))

    def test_scaler_fits_real_training_only_and_modes_pair(self):
        """改变验证温度不会影响训练标准化参数，双通道不改变温度通道。"""
        batch = self.api.build_windows(make_windows())
        tr, va = self.api.group_folds(batch)[0]
        config = {'A': False, 'S': True, 'T': False, 'smote_amount': 100}
        a = self.api.prepare_training(batch.subset(tr), config, 'temp_only', 123)
        b = self.api.prepare_training(batch.subset(tr), config, 'temp_rate', 123)
        np.testing.assert_allclose(a[0][:,:,0], b[0][:,:,0], atol=1e-6)
        self.assertEqual(len(a[1]), 2*len(tr))
        self.assertAlmostEqual(b[2].mean_[0], batch.temperatures[tr].mean())
        self.assertNotAlmostEqual(b[2].mean_[1], 999)
        transformed = self.api.transform_windows(batch.subset(va), b[2], 'temp_rate')
        self.assertEqual(transformed.shape, (len(va),48,2))

    def test_custom_and_standard_cleaning_are_distinct(self):
        """类内近邻更近时，标准Tomek不会误删异类互近邻样本。"""
        flat = pd.DataFrame({'sSowsNo': ['a','b','c','d'], **{f'feature{i}':[0.,.1,10.,10.1] for i in range(1,49)}, 'isEstrus':[0,0,1,1]})
        custom, _ = self.api.augment_flat(flat, {'T': True, 'tomek_mode':'custom'}, 10)
        standard, _ = self.api.augment_flat(flat, {'T': True, 'tomek_mode':'standard'}, 10)
        self.assertEqual(len(custom), 3)
        self.assertEqual(len(standard), 4)

    def test_prediction_export_joins_keys_and_excludes_features(self):
        """按键合并乱序概率，保存原目录，编号前导零可往返。"""
        batch = self.api.build_windows(make_windows())
        records = pd.DataFrame({'sSowsNo_split': batch.metadata.sSowsNo_split.iloc[::-1], 'y_prob': np.arange(20)/20, 'fold': 1})
        with test_directory() as directory:
            source = Path(directory)/'test.xlsx'
            source.write_bytes(b'original')
            first = self.api.export_predictions(batch, records, source, 'E8_test', 766, 'run', 'BiLSTM')
            second = self.api.export_predictions(batch, records, source, 'E8_test', 766, 'run', 'BiLSTM')
            self.assertEqual(first.parent, source.parent)
            self.assertNotEqual(first, second)
            self.assertEqual(source.read_bytes(), b'original')
            saved = pd.read_excel(first, dtype={'sSowsNo':str,'sEarTagCode':str,'sSowsNo_split':str})
            self.assertEqual(len(saved), 20)
            self.assertEqual(saved.iloc[0].sSowsNo, '0000')
            self.assertAlmostEqual(saved.iloc[0].y_prob, .95)
            self.assertFalse(any('Temperature' in c or 'temperatureRate' in c or c=='iStep' for c in saved.columns))
            self.assertIn('source_file', saved)
            self.assertIn('window_start', saved)
            self.assertIn('window_end', saved)

    def test_missing_or_duplicate_predictions_are_rejected(self):
        """缺预测或重复键时禁止生成误导的结果表。"""
        batch = self.api.build_windows(make_windows())
        records = pd.DataFrame({'sSowsNo_split':batch.metadata.sSowsNo_split, 'y_prob':.2})
        with test_directory() as directory:
            for bad in [records.iloc[:-1], pd.concat([records,records.iloc[:1]])]:
                with self.assertRaises(ValueError):
                    self.api.export_predictions(batch,bad,Path(directory)/'test.xlsx','E',1,'r','BiLSTM')


if __name__ == '__main__':
    unittest.main()
