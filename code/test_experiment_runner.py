"""验证实验矩阵、缓存边界和真实训练导出接口。"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_experiment_protocol import make_windows, test_directory


class RunnerTests(unittest.TestCase):
    """在微型真实数据上检查调度和预测生命周期。"""

    def setUp(self):
        """载入待实现调度器，并将临时文件限制在测试工作区。"""
        self.assertIsNotNone(importlib.util.find_spec('experiment_runner'), '缺少实验调度器')
        import experiment_runner
        self.api = experiment_runner
        self.root = Path(self.enterContext(test_directory()))
        self.data_dir = self.root/'1'
        self.data_dir.mkdir()
        make_windows().to_excel(self.data_dir/'train_val_df.xlsx',index=False)
        test = make_windows(sows=2)
        test['sSowsNo'] = 'T'+test.sSowsNo
        test['sSowsNo_split'] = 'T'+test.sSowsNo_split
        test.to_excel(self.data_dir/'test.xlsx',index=False)

    def test_matrix_sizes_and_fixed_model_comparison(self):
        """采样×特征完整，标准清理4项，模型对照均四层64。"""
        self.assertEqual(len(self.api.stage_configs('E1')),16)
        self.assertEqual(len(self.api.stage_configs('E2')),4)
        self.assertEqual(len(self.api.stage_configs('E3')),5)
        self.assertEqual(len(self.api.stage_configs('E4')),10)
        configs = self.api.stage_configs('E7')
        self.assertEqual(len(configs),4)
        self.assertTrue(all(c['hidden_sizes']==[64]*4 for c in configs))
        self.assertTrue(all(c['smote_amount']==800 for c in configs))

    def test_effective_key_ignores_display_name_but_not_protocol(self):
        """跨阶段相同训练可以复用，清理规则及特征变化必须改变缓存键。"""
        config = self.api.stage_configs('E1')[0]
        self.assertEqual(self.api.config_key(config), self.api.config_key({**config,'name':'another','experiment_id':'E99'}))
        self.assertNotEqual(self.api.config_key(config),self.api.config_key({**config,'feature_mode':'temp_rate'}))
        ast = self.api.stage_configs('E1')[-1]
        self.assertNotEqual(self.api.config_key(ast),self.api.config_key({**ast,'tomek_mode':'standard'}))

    def test_cv_exports_oof_without_test_evaluation_and_resumes(self):
        """真实五折短训练生成完整OOF；缓存复用时不能重新训练或读取测试概率。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1,split_index=7)
        config = {**self.api.stage_configs('E1')[0], 'hidden_sizes':[4], 'batch_size':8}
        with patch.object(runner,'evaluate_final',side_effect=AssertionError('开发阶段触碰测试')):
            result = runner.evaluate_cv(config,766)
        pred = pd.read_excel(result['prediction_path'],dtype={'sSowsNo_split':str})
        self.assertEqual(len(pred),20)
        self.assertEqual(pred.sSowsNo_split.nunique(),20)
        self.assertTrue(pred.data_split_index.eq(7).all())
        self.assertTrue(Path(result['prediction_path']).parent.samefile(self.data_dir))
        with patch.object(runner,'fit',side_effect=AssertionError('缓存未生效')):
            resumed = runner.evaluate_cv(config,766)
        self.assertEqual(result['prediction_path'],resumed['prediction_path'])
        self.assertEqual(result['metrics'],resumed['metrics'])

    def test_final_requires_frozen_config_and_reloads_probabilities(self):
        """最终输出必须来自冻结配置，模型重新加载后概率一致。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        config = {**self.api.stage_configs('E1')[0], 'hidden_sizes':[4], 'batch_size':8}
        with self.assertRaisesRegex(ValueError,'冻结'):
            runner.evaluate_final(config,766)
        runner.evaluate_cv(config,766)
        frozen = runner.freeze_config(config,[766])
        result = runner.evaluate_final(frozen,766)
        pred = pd.read_excel(result['prediction_path'])
        self.assertEqual(len(pred),4)
        restored = runner.reload_probabilities(result['model_dir'],runner.test)
        np.testing.assert_allclose(restored,pred.y_prob,atol=1e-7)

    def test_changed_source_rejects_resume(self):
        """输入文件变更后不能沿用原运行缓存。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        changed = make_windows()
        changed.loc[0,'iTemperature'] += 1
        changed.to_excel(self.data_dir/'train_val_df.xlsx',index=False)
        with self.assertRaisesRegex(ValueError,'协议|数据|身份'):
            self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1,run_dir=runner.run_dir)

    def test_reports_use_all_seed_predictions(self):
        """报告读取已完成标记中的全部种子，而非挑选最优种子。"""
        self.assertIsNotNone(importlib.util.find_spec('experiment_reports'), '缺少实验报告生成器')
        from experiment_reports import build_reports
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        config = {**self.api.stage_configs('E1')[0], 'hidden_sizes':[4], 'batch_size':8}
        frozen = runner.freeze_config(config,[766],calibrate=True)
        for seed in [766,929]:
            runner.evaluate_final(frozen,seed)
        runner.write_final_summary()
        build_reports(runner.run_dir)
        self.assertTrue((runner.run_dir/'figures'/'final_roc_pr.png').is_file())
        self.assertTrue((runner.run_dir/'figures'/'final_confusion.png').is_file())
        self.assertTrue((runner.run_dir/'figures'/'training_validation_loss.png').is_file())
        self.assertTrue((runner.run_dir/'report.md').is_file())

    def test_bayes_uses_oof_only_and_honors_warm_start(self):
        """小搜索预算跑真实五折，并检查首候选及搜索跟踪文件。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        warm = self.api.stage_configs('E4')[0]
        with patch.object(runner,'evaluate_final',side_effect=AssertionError('搜索不允许测试')):
            results = runner.bayesian_search('custom',warm,n_calls=1)
        self.assertEqual(results[0]['config']['smote_amount'],100)
        self.assertTrue((runner.run_dir/'E5_search_trace.xlsx').exists())

    def test_modified_export_cannot_change_cached_validation_or_threshold(self):
        """手工修改导出Excel后必须拒绝缓存复用，不能静默改变选模或阈值。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        config = {**self.api.stage_configs('E1')[0], 'hidden_sizes':[4], 'batch_size':8}
        result = runner.evaluate_cv(config,766)
        path = Path(result['prediction_path'])
        path.write_bytes(path.read_bytes()+b'edited')
        with self.assertRaisesRegex(ValueError,'缓存|指纹'):
            runner.evaluate_cv(config,766)

    def test_missing_model_is_not_a_completed_cache(self):
        """模型权重丢失时完整标记不能继续冒充有效训练结果。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        config = {**self.api.stage_configs('E1')[0], 'hidden_sizes':[4], 'batch_size':8}
        result = runner.evaluate_cv(config,766)
        (Path(result['model_dir'])/'fold_1'/'model.pth').unlink()
        with self.assertRaisesRegex(ValueError,'缓存|缺失'):
            runner.evaluate_cv(config,766)

    def test_threshold_calibration_uses_internal_oof_and_authoritative_labels(self):
        """冻结阈值不读取导出的Excel，避免可编辑表格成为标签和概率真源。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        config = {**self.api.stage_configs('E1')[0], 'hidden_sizes':[4], 'batch_size':8}
        runner.evaluate_cv(config,766)
        with patch.object(pd,'read_excel',side_effect=AssertionError('不得读取导出Excel校准')):
            frozen = runner.freeze_config(config,[766],calibrate=True)
        self.assertTrue(0.05 <= frozen['calibrated_threshold'] <= .95)

    def test_missing_export_is_restored_without_retraining(self):
        """仅导出表丢失时由已校验的内部概率恢复，避免重新训练五折。"""
        runner = self.api.ExperimentRunner(self.data_dir,self.root/'result',profile='smoke',epochs=1)
        config = {**self.api.stage_configs('E1')[0], 'hidden_sizes':[4], 'batch_size':8}
        result = runner.evaluate_cv(config,766)
        original = pd.read_excel(result['prediction_path'])
        Path(result['prediction_path']).unlink()
        with patch.object(runner,'fit',side_effect=AssertionError('不应重训')):
            restored = runner.evaluate_cv(config,766)
        pd.testing.assert_frame_equal(original,pd.read_excel(restored['prediction_path']))


if __name__ == '__main__':
    unittest.main()
