"""实验共用的数据协议：按窗口建样本、按母猪划分、训练内增强和可追溯导出。"""

import contextlib
import hashlib
import io
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = ['sEarTagCode', 'sSowsNo', 'sSowsNo_split']
META_COLUMNS = ID_COLUMNS + ['sBrand', 'dBreedDate', 'dWeanDate',
                            'window_start', 'window_end', 'tLastUploadTime']
TEMP_COLUMNS = [f'feature{i}' for i in range(1, 49)]


@dataclass
class WindowBatch:
    """保存顺序严格一致的窗口温度(N,48)、标签(N,)和真实样本元数据。"""

    temperatures: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame

    def subset(self, indices):
        """按整数位置提取子集；同步复制标签和元数据，不改变原始批次。"""
        return WindowBatch(self.temperatures[indices].copy(), self.labels[indices].copy(),
                           self.metadata.iloc[indices].reset_index(drop=True).copy())


def build_windows(data):
    """将小时长表转为独立48小时窗口；校验编号、时间和数值，不筛掉样本或写文件。"""
    required = {'sSowsNo', 'sSowsNo_split', 'tLastUploadTime', 'iTemperature', 'isEstrus'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'缺少窗口列: {sorted(missing)}')
    if data.empty:
        raise ValueError('实验数据没有窗口')
    frame = data.copy()
    for col in ID_COLUMNS:
        if col in frame:
            if frame[col].isna().any():
                raise ValueError(f'编号列 {col} 存在缺失')
            frame[col] = frame[col].astype(str)
    frame['tLastUploadTime'] = pd.to_datetime(frame['tLastUploadTime'])
    if not frame.isEstrus.isin([0, 1]).all():
        raise ValueError('isEstrus 必须为0或1')
    # 沿用已确认的母猪类别：同一阳性母猪的多个发情周期继承该类别。
    positive_sows = set(frame.loc[frame.isEstrus.eq(1), 'sSowsNo'])
    temperatures, labels, metadata = [], [], []
    for sample_id, group in frame.groupby('sSowsNo_split', sort=True):
        group = group.sort_values('tLastUploadTime')
        times = group.tLastUploadTime
        if len(group) != 48 or times.isna().any() or not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all() or not times.eq(times.dt.floor('h')).all():
            raise ValueError(f'样本 {sample_id} 必须包含48个连续、不重复的整点小时')
        for col in ID_COLUMNS:
            if col in group and group[col].nunique() != 1:
                raise ValueError(f'样本 {sample_id} 的 {col} 不一致')
        values = group.iTemperature.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'样本 {sample_id} 温度存在非有限值')
        last = group.iloc[-1]
        meta = {col: last[col] for col in META_COLUMNS if col in group}
        meta.update(window_start=times.iloc[0], window_end=times.iloc[-1])
        temperatures.append(values)
        labels.append(int(last.sSowsNo in positive_sows))
        metadata.append(meta)
    return WindowBatch(np.asarray(temperatures), np.asarray(labels, dtype=int), pd.DataFrame(metadata))


def group_folds(batch, n_splits=5, seed=54):
    """对唯一母猪分层K折，返回窗口位置索引；同一母猪的全部周期留在同一折。"""
    sow_labels = pd.DataFrame({'sow': batch.metadata.sSowsNo, 'y': batch.labels}).groupby('sow', sort=True).y.max()
    if len(sow_labels.value_counts()) != 2 or sow_labels.value_counts().min() < n_splits:
        raise ValueError(f'每类至少需要 {n_splits} 头母猪，不能构造分层五折')
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = []
    for train, val in splitter.split(sow_labels.index, sow_labels.values):
        tr = np.flatnonzero(batch.metadata.sSowsNo.isin(sow_labels.index[train]))
        va = np.flatnonzero(batch.metadata.sSowsNo.isin(sow_labels.index[val]))
        result.append((tr, va))
    return result


def channels(temperatures, feature_mode):
    """由温度构造单通道或温度加窗口内差分；首小时差分固定为0。"""
    values = np.asarray(temperatures, dtype=float)
    if values.ndim != 2 or values.shape[1] != 48:
        raise ValueError('温度矩阵形状必须为(N,48)')
    if feature_mode == 'temp_only':
        return values[:, :, None]
    if feature_mode != 'temp_rate':
        raise ValueError(f'未知特征模式: {feature_mode}')
    rates = np.diff(values, axis=1, prepend=values[:, :1])
    return np.stack([values, rates], axis=-1)


def flat_windows(batch):
    """生成旧采样函数需要的50列温度表；首列存样本编号，真实母猪信息独立保存。"""
    flat = pd.DataFrame(batch.temperatures, columns=TEMP_COLUMNS)
    flat.insert(0, 'sSowsNo', batch.metadata.sSowsNo_split.to_numpy())
    flat['isEstrus'] = batch.labels
    return flat


def augment_flat(flat, config, seed):
    """仅增强传入训练表，依次执行A/S/T并返回类别计数；恢复调用者的随机状态。"""
    import sow_estrus_LSTM_Function as legacy
    from imblearn.under_sampling import TomekLinks
    frame = flat.copy()
    if set(frame.isEstrus.unique()) != {0, 1}:
        raise ValueError('训练数据必须同时包含正负类')
    counts = []
    state, py_state = np.random.get_state(), random.getstate()
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        for stage in ['input', 'A', 'S', 'T']:
            with contextlib.redirect_stdout(io.StringIO()):
                if stage == 'A' and config.get('A', False):
                    positive, negative = frame[frame.isEstrus.eq(1)], frame[frame.isEstrus.eq(0)]
                    if len(positive) < 2:
                        raise ValueError('ADASYN至少需要两个正类训练样本')
                    frame = legacy.ADASYN(.9, 1, positive, negative, k=min(7, len(frame)-1))
                elif stage == 'S' and config.get('S', False):
                    frame = legacy.SMOTE(frame, amount_oversampling=int(config.get('smote_amount', 800)), k=7)
                elif stage == 'T' and config.get('T', False):
                    mode = config.get('tomek_mode', 'custom')
                    if mode == 'custom':
                        frame = legacy.TomekLinked(frame, k=1)
                    elif mode == 'standard':
                        sampler = TomekLinks(sampling_strategy=[0])
                        sampler.fit_resample(frame[TEMP_COLUMNS].to_numpy(), frame.isEstrus.to_numpy())
                        frame = frame.iloc[sampler.sample_indices_].reset_index(drop=True)
                    else:
                        raise ValueError(f'未知清理方法: {mode}')
            counts.append({'stage': stage, 'negative': int(frame.isEstrus.eq(0).sum()),
                           'positive': int(frame.isEstrus.eq(1).sum()), 'total': len(frame)})
        if set(frame.isEstrus.unique()) != {0, 1}:
            raise ValueError('采样后有类别被全部删除')
        return frame, counts
    finally:
        np.random.set_state(state)
        random.setstate(py_state)


def prepare_training(batch, config, feature_mode, seed):
    """在真实训练窗口拟合scaler，增强温度后重算差分；返回X、y、scaler和采样计数。"""
    real = channels(batch.temperatures, feature_mode)
    scaler = StandardScaler().fit(real.reshape(-1, real.shape[-1]))
    flat, counts = augment_flat(flat_windows(batch), config, seed)
    raw = channels(flat[TEMP_COLUMNS].to_numpy(), feature_mode)
    X = scaler.transform(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape).astype(np.float32)
    return X, flat.isEstrus.to_numpy(dtype=int), scaler, counts


def transform_windows(batch, scaler, feature_mode):
    """用训练scaler转换真实验证/测试窗口，不重新拟合或增强。"""
    raw = channels(batch.temperatures, feature_mode)
    return scaler.transform(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape).astype(np.float32)


def legacy_tensors(data, feature_mode, scaler=None):
    """兼容旧长表及展开表接口，按列名提取温度并统一差分，返回X、y、scaler。"""
    if 'iTemperature' in data:
        frame = data.copy()
        if 'sSowsNo_split' not in frame:
            frame['sSowsNo_split'] = frame.sSowsNo.astype(str)
        batch = build_windows(frame)
        temps, labels = batch.temperatures, batch.labels
    else:
        candidates = [TEMP_COLUMNS, [f'iTemperature_{i}' for i in range(1,49)],
                      [f'temperature_{i}' for i in range(1,49)]]
        cols = next((cols for cols in candidates if set(cols).issubset(data.columns)), None)
        if cols is None:
            raise ValueError('无法按列名识别48个温度特征')
        temps, labels = data[cols].to_numpy(dtype=float), data.isEstrus.to_numpy(dtype=int)
    raw = channels(temps, feature_mode)
    if scaler is None:
        scaler = StandardScaler().fit(raw.reshape(-1, raw.shape[-1]))
    X = scaler.transform(raw.reshape(-1,raw.shape[-1])).reshape(raw.shape).astype(np.float32)
    return X, labels, scaler


def binary_metrics(labels, probabilities, threshold=.5):
    """由真实标签和概率计算二分类指标；AUC/AP不依赖分类阈值。"""
    labels, probs = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or len(labels) != len(probs) or not np.isfinite(probs).all():
        raise ValueError('标签与概率长度不匹配、为空或包含非有限值')
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'F1-Score':float(f1_score(labels,pred,zero_division=0)),
            'Accuracy':float(accuracy_score(labels,pred)),
            'Precision':float(precision_score(labels,pred,zero_division=0)),
            'Recall':float(recall_score(labels,pred,zero_division=0)),
            'Specificity':float(tn/(tn+fp)) if tn+fp else 0.,
            'AUC':float(roc_auc_score(labels,probs)) if len(np.unique(labels)) == 2 else float('nan'),
            'AP':float(average_precision_score(labels,probs)) if labels.sum() else 0.,
            'MCC':float(matthews_corrcoef(labels,pred)), 'Threshold':float(threshold),
            'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}


def file_sha256(path):
    """分块读取文件并返回SHA256，供输入和缓存身份验证使用，不写文件。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_predictions(batch, records, source_path, experiment_id, seed, run_id,
                       model_name, dataset='Independent_Test', threshold=.5,
                       calibrated_threshold=None, split_index=None):
    """按样本键关联概率，以白名单导出原数据同目录Excel；返回新路径，绝不覆盖已有文件。"""
    source = Path(source_path).resolve()
    key = 'sSowsNo_split'
    if key not in records or 'y_prob' not in records or records[key].duplicated().any():
        raise ValueError('预测缺少样本键/概率，或样本键重复')
    if len(records) != len(batch.labels) or set(records[key]) != set(batch.metadata[key]):
        raise ValueError('预测样本集合与原始窗口不完全一致')
    probs = records.y_prob.to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any():
        raise ValueError('预测概率必须是[0,1]内的有限数值')
    frame = batch.metadata[[c for c in META_COLUMNS if c in batch.metadata]].copy()
    frame['isEstrus'] = batch.labels
    selected = [key, 'y_prob'] + (['fold'] if 'fold' in records else [])
    frame = frame.merge(records[selected], on=key, how='left', validate='one_to_one', sort=False)
    frame['y_pred'] = (frame.y_prob >= threshold).astype(int)
    frame['threshold'] = float(threshold)
    if calibrated_threshold is not None:
        frame['y_pred_calibrated'] = (frame.y_prob >= calibrated_threshold).astype(int)
        frame['calibrated_threshold'] = float(calibrated_threshold)
    if 'fold' not in frame:
        frame['fold'] = pd.NA
    for col,value in {'source_file':str(source),'data_split_index':split_index if split_index is not None else source.parent.name,
                      'experiment_id':experiment_id,'run_id':run_id,'model_name':model_name,
                      'seed':int(seed),'dataset':dataset}.items():
        frame[col] = value
    suffix = 'oof' if dataset == 'Validation' else 'predictions'
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(experiment_id))
    safe_run = re.sub(r'[^a-zA-Z0-9_-]', '_', str(run_id))
    stem = f'{source.stem}__{suffix}__{safe_id}__seed{seed}__{safe_run}'
    attempt = 0
    while True:
        destination = source.parent / f'{stem}{"__"+str(attempt) if attempt else ""}.xlsx'
        try:
            stream = destination.open('xb')
            break
        except FileExistsError:
            attempt += 1
    try:
        with stream, pd.ExcelWriter(stream, engine='openpyxl') as writer:
            frame.to_excel(writer, index=False, sheet_name='predictions')
            sheet = writer.sheets['predictions']
            for col in ID_COLUMNS:
                if col in frame:
                    position = frame.columns.get_loc(col)+1
                    for row in range(2,len(frame)+2):
                        sheet.cell(row,position).data_type = 's'
                        sheet.cell(row,position).number_format = '@'
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination
