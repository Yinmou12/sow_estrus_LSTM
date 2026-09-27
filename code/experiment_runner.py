"""第1套数据实验执行器：可恢复五折验证、配置冻结、最终测试及原目录预测导出。"""

import argparse
import hashlib
import json
import importlib.metadata
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiment_configs import (BASE_CONFIG, FINAL_SEEDS, SEARCH_SPACE, TRAIN_SEEDS,
                                config_key, effective_config, named_config, stage_configs)
from experiment_data import (ID_COLUMNS, binary_metrics, build_windows, export_predictions,
                             file_sha256, group_folds, prepare_training, transform_windows)
from sow_estrus_LSTM_train import (EstrusDataset, __train_info as TrainingInfo,
                                   find_best_threshold, get_model)

PROTOCOL_VERSION = 'window48_group_sow_v1'
METRICS = ['F1-Score','Accuracy','Precision','Recall','Specificity','AUC','AP','MCC']
MODEL_ARTIFACTS = ['model.pth','scaler.joblib','config.json','fit_info.json','history.csv','augmentation_counts.csv']


def write_json(path, value):
    """先写同目录临时文件再原子替换JSON，避免中断留下不完整的完成标记。"""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    """读取UTF-8 JSON配置或完成标记并返回原生对象。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def artifact_hashes(paths):
    """为完成任务的必需产物保存内容指纹，防止损坏模型或手工修改预测被静默复用。"""
    return {str(Path(path).resolve()):file_sha256(path) for path in paths}


def verify_artifacts(hashes,allow_missing=None):
    """校验缓存产物；仅允许明确指定的导出文件缺失，其余缺失或被改动立即报错。"""
    if not hashes:
        raise ValueError('缓存缺少产物指纹，请创建新的运行目录')
    for filename,expected in hashes.items():
        path = Path(filename)
        if not path.exists() and allow_missing is not None and path == Path(allow_missing):
            continue
        if not path.exists():
            raise ValueError(f'缓存产物缺失: {path}')
        if file_sha256(path) != expected:
            raise ValueError(f'缓存产物指纹不一致: {path}')


def set_training_seed(seed):
    """重置模型和DataLoader使用的随机源，配置顺序不影响独立训练结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config, device):
    """显式构造实际网络配置；GRU只接受等宽隐藏层，禁止记录与实际结构不一致。"""
    info = TrainingInfo()
    for key,value in effective_config(config).items():
        if hasattr(info,key):
            setattr(info,key,value)
    info.num_layers = len(config['hidden_sizes'])
    info.hidden_sizes = list(config['hidden_sizes'])
    info.layer_hidden_size = config['hidden_sizes'][0]
    if config['model_name'] == 'EstrusGRU' and len(set(config['hidden_sizes'])) != 1:
        raise ValueError('现有GRU仅支持等宽隐藏层')
    return get_model(info,device,input_size=1 if config['feature_mode']=='temp_only' else 2)


def predict(model, X, batch_size, device):
    """按输入顺序输出概率，评估期间关闭梯度与Dropout。"""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0,len(X),batch_size):
            tensor = torch.as_tensor(X[start:start+batch_size],dtype=torch.float32,device=device)
            chunks.append(model(tensor).detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks)


def rank_results(results):
    """按跨种子OOF均值F1、AUC、Recall稳定排序，完全平局保留原候选顺序。"""
    return sorted(results,key=lambda result:tuple(-result['mean'][key] for key in ['F1-Score','AUC','Recall']))


class ExperimentRunner:
    """管理一次可恢复实验；来源目录只新增预测文件，模型和中间记录写结果目录。"""

    def __init__(self,data_dir,result_root,*,profile='full',epochs=100,run_dir=None,cv_seed=54,
                 n_splits=5,train_seeds=None,final_seeds=None,split_index=None):
        """读取已冻结输入并校验运行身份；resume时拒绝数据、代码或协议变化。"""
        self.data_dir = Path(data_dir).resolve()
        self.sources = {'train':self.data_dir/'train_val_df.xlsx','test':self.data_dir/'test.xlsx'}
        self.profile,self.epochs,self.cv_seed = profile,int(epochs),int(cv_seed)
        self.split_index = split_index if split_index is not None else self.data_dir.name
        self.train_seeds = list(train_seeds or TRAIN_SEEDS)
        self.final_seeds = list(final_seeds or FINAL_SEEDS)
        if self.epochs < 1:
            raise ValueError('epochs必须为正整数')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 小型循环网络避免CPU过度并行；此设置也记录在运行身份中。
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        source_hashes = {key:file_sha256(path) for key,path in self.sources.items()}
        code_names = ['experiment_runner.py','experiment_data.py','experiment_configs.py',
                      'sow_estrus_LSTM_Function.py','sow_estrus_LSTM_train.py','lstm_model.py']
        code_hashes = {name:file_sha256(Path(__file__).parent/name) for name in code_names}
        self.identity = {'protocol':PROTOCOL_VERSION,'data_dir':str(self.data_dir),'data':source_hashes,
                         'code':code_hashes,'profile':profile,'epochs':self.epochs,'cv_seed':cv_seed,
                         'n_splits':n_splits,'train_seeds':self.train_seeds,'final_seeds':self.final_seeds,
                         'torch_version':str(torch.__version__),'device':str(self.device),'cpu_threads':torch.get_num_threads(),
                         'split_index':self.split_index,
                         'packages':{name:importlib.metadata.version(name) for name in
                                     ['numpy','pandas','scikit-learn','imbalanced-learn','scikit-optimize','openpyxl']},
                         'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
                         'gpu':torch.cuda.get_device_name(0) if self.device.type=='cuda' else None}
        if run_dir is None:
            self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.run_dir = Path(result_root).resolve()/f'protocol_v1_{profile}_{self.run_id}'
            self.run_dir.mkdir(parents=True,exist_ok=False)
            write_json(self.run_dir/'manifest.json',{'run_id':self.run_id,'identity':self.identity})
        else:
            self.run_dir = Path(run_dir).resolve()
            saved = read_json(self.run_dir/'manifest.json')
            if saved['identity'] != self.identity:
                raise ValueError('恢复运行的协议/数据/代码身份不一致，请创建新运行')
            self.run_id = saved['run_id']
        dtype = {col:str for col in ID_COLUMNS}
        self.train = build_windows(pd.read_excel(self.sources['train'],dtype=dtype))
        self.test = build_windows(pd.read_excel(self.sources['test'],dtype=dtype))
        if set(self.train.metadata.sSowsNo)&set(self.test.metadata.sSowsNo):
            raise ValueError('训练/验证与独立测试存在母猪交叉')
        if set(self.train.metadata.sSowsNo_split)&set(self.test.metadata.sSowsNo_split):
            raise ValueError('训练/验证与测试存在重复样本编号')
        self.folds = group_folds(self.train,n_splits=n_splits,seed=cv_seed)
        self.audit()

    def log(self,message,**details):
        """追加一条带时间的运行事件，并刷新控制台以便外部观察长实验进度。"""
        event = {'time':datetime.now().isoformat(timespec='seconds'),'message':message,**details}
        with (self.run_dir/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,ensure_ascii=False)+'\n')
        print(json.dumps(event,ensure_ascii=False),flush=True)

    def audit(self):
        """保存窗口统计和固定折清单；只检查测试元数据，不预测测试标签。"""
        rows = []
        for label,batch in [('train_val',self.train),('test',self.test)]:
            rows.append({'dataset':label,'sows':batch.metadata.sSowsNo.nunique(),'windows':len(batch.labels),
                         'positive':int(batch.labels.sum()),'negative':int((batch.labels==0).sum())})
        pd.DataFrame(rows).to_excel(self.run_dir/'dataset_counts.xlsx',index=False)
        assignment = self.train.metadata.copy()
        assignment['isEstrus'] = self.train.labels
        assignment['fold'] = 0
        for fold,(_,val) in enumerate(self.folds,1):
            assignment.loc[val,'fold'] = fold
        if (assignment.fold==0).any():
            raise RuntimeError('五折没有完整覆盖所有窗口')
        assignment.to_excel(self.run_dir/'fold_manifest.xlsx',index=False)
        self.log('data_audit',counts=rows,device=str(self.device))
        return rows

    def fit(self,config,train_batch,val_batch,seed,fold,model_dir,epochs=None):
        """执行一个训练任务并保存模型、scaler、历史和采样计数；验证只负责早停。"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True,exist_ok=True)
        started = time.perf_counter()
        # 增强随机源与网络随机源独立，同一窗口/种子在特征消融中共享温度增强。
        augmentation_seed = int(seed)*1000+int(fold)
        X,y,scaler,counts = prepare_training(train_batch,config,config['feature_mode'],augmentation_seed)
        X_val = transform_windows(val_batch,scaler,config['feature_mode']) if val_batch is not None else None
        set_training_seed(int(seed)*1000+int(fold)+10000000)
        model = build_model(config,self.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=config['lr_patience']) if val_batch is not None else None
        loader = DataLoader(EstrusDataset(X,y),batch_size=config['batch_size'],shuffle=True,drop_last=False)
        criterion = torch.nn.BCELoss()
        history,best_loss,best_epoch,stale = [],float('inf'),0,0
        max_epochs = int(epochs or self.epochs)
        self.log('fit_start',config=config_key(config),seed=seed,fold=fold,samples=len(y),epochs=max_epochs)
        for epoch in range(1,max_epochs+1):
            model.train()
            total,seen = 0.,0
            for bx,by in loader:
                bx,by = bx.to(self.device),by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(bx),by)
                if not torch.isfinite(loss):
                    raise RuntimeError('训练损失出现非有限值')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                total += float(loss.item())*len(by)
                seen += len(by)
            row = {'epoch':epoch,'train_loss':total/seen,'learning_rate':optimizer.param_groups[0]['lr']}
            if val_batch is not None:
                probs = predict(model,X_val,config['batch_size'],self.device)
                val_loss = float(criterion(torch.tensor(probs,dtype=torch.float32),torch.tensor(val_batch.labels,dtype=torch.float32)).item())
                row['val_loss'] = val_loss
                if val_loss < best_loss:
                    best_loss,best_epoch,stale = val_loss,epoch,0
                    torch.save(model.state_dict(),model_dir/'model.pth')
                else:
                    stale += 1
                scheduler.step(val_loss)
            else:
                best_epoch = epoch
            history.append(row)
            if epoch % 10 == 0:
                self.log('fit_epoch',config=config_key(config),seed=seed,fold=fold,**row)
            if val_batch is not None and stale >= config['early_patience']:
                break
        if val_batch is None:
            torch.save(model.state_dict(),model_dir/'model.pth')
        else:
            model.load_state_dict(torch.load(model_dir/'model.pth',map_location=self.device,weights_only=True))
        joblib.dump(scaler,model_dir/'scaler.joblib')
        pd.DataFrame(history).to_csv(model_dir/'history.csv',index=False)
        pd.DataFrame(counts).to_csv(model_dir/'augmentation_counts.csv',index=False)
        write_json(model_dir/'config.json',config)
        info = {'best_epoch':int(best_epoch),'epochs_executed':len(history),
                'train_seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters()),
                'augmentation_seed':augmentation_seed,'model_seed':int(seed)*1000+int(fold)+10000000}
        write_json(model_dir/'fit_info.json',info)
        self.log('fit_complete',config=config_key(config),seed=seed,fold=fold,**info)
        return model,scaler,info

    def reload_probabilities(self,model_dir,batch):
        """重新读取保存的配置、权重和scaler，返回给定窗口的概率以验证复现。"""
        model_dir = Path(model_dir)
        config = read_json(model_dir/'config.json')
        scaler = joblib.load(model_dir/'scaler.joblib')
        model = build_model(config,self.device)
        model.load_state_dict(torch.load(model_dir/'model.pth',map_location=self.device,weights_only=True))
        return predict(model,transform_windows(batch,scaler,config['feature_mode']),config['batch_size'],self.device)

    def restore_export(self,result,batch,source,dataset):
        """在模型与内部概率校验通过后重建丢失的Excel，更新路径及指纹，不重新训练。"""
        probabilities = np.load(result['probabilities_path'],allow_pickle=False)
        records = pd.DataFrame({'sSowsNo_split':batch.metadata.sSowsNo_split,'y_prob':probabilities})
        if dataset=='Validation':
            records['fold'] = 0
            for fold,(_,val) in enumerate(self.folds,1):
                records.loc[val,'fold'] = fold
        config = result['config']
        path = export_predictions(batch,records,source,('CV_' if dataset=='Validation' else 'FINAL_')+result['config_key'],
                                  result['seed'],self.run_id,config['model_name'],dataset=dataset,
                                  calibrated_threshold=config.get('calibrated_threshold') if dataset!='Validation' else None,
                                  split_index=self.split_index)
        result['artifacts'].pop(result['prediction_path'])
        result['prediction_path'] = str(path)
        result['artifacts'].update(artifact_hashes([path]))
        return result

    def evaluate_cv(self,config,seed):
        """训练或恢复五折模型，汇总一次完整OOF评估并在训练文件同目录导出预测。"""
        config = {**BASE_CONFIG,**config}
        key = config_key(config)
        cache = self.run_dir/'cv'/key/f'seed_{seed}'
        cache.mkdir(parents=True,exist_ok=True)
        marker = cache/'complete.json'
        if marker.exists():
            result = read_json(marker)
            verify_artifacts(result['artifacts'],allow_missing=result['prediction_path'])
            for fold in result['folds']:
                verify_artifacts(fold['artifacts'])
            if not Path(result['prediction_path']).exists():
                result = self.restore_export(result,self.train,self.sources['train'],'Validation')
                write_json(marker,result)
            return result
        predictions,details = [],[]
        for fold,(train_idx,val_idx) in enumerate(self.folds,1):
            folder = cache/f'fold_{fold}'
            checkpoint = folder/'complete.json'
            if checkpoint.exists():
                detail = read_json(checkpoint)
                verify_artifacts(detail['artifacts'])
                probs = np.load(folder/'probabilities.npy',allow_pickle=False)
            else:
                train,val = self.train.subset(train_idx),self.train.subset(val_idx)
                model,scaler,info = self.fit(config,train,val,seed,fold,folder)
                probs = predict(model,transform_windows(val,scaler,config['feature_mode']),config['batch_size'],self.device)
                np.save(folder/'probabilities.npy',probs,allow_pickle=False)
                detail = {'fold':fold,**info,'metrics':binary_metrics(val.labels,probs)}
                detail['artifacts'] = artifact_hashes([folder/name for name in MODEL_ARTIFACTS+['probabilities.npy']])
                write_json(checkpoint,detail)
            if len(probs)!=len(val_idx):
                raise ValueError('缓存验证概率长度错误')
            predictions.append(pd.DataFrame({'sSowsNo_split':self.train.metadata.iloc[val_idx].sSowsNo_split.to_numpy(),
                                             'y_prob':probs,'fold':fold}))
            details.append(detail)
        records = pd.concat(predictions,ignore_index=True)
        ordered = self.train.metadata[['sSowsNo_split']].merge(records,on='sSowsNo_split',validate='one_to_one',sort=False)
        metrics = binary_metrics(self.train.labels,ordered.y_prob.to_numpy())
        np.save(cache/'oof_probabilities.npy',ordered.y_prob.to_numpy(),allow_pickle=False)
        path = export_predictions(self.train,records,self.sources['train'],'CV_'+key,seed,self.run_id,
                                  config['model_name'],dataset='Validation',split_index=self.split_index)
        result = {'config_key':key,'config':config,'seed':int(seed),'metrics':metrics,'folds':details,
                  'prediction_path':str(path),'model_dir':str(cache),'probabilities_path':str(cache/'oof_probabilities.npy'),
                  'artifacts':artifact_hashes([path,cache/'oof_probabilities.npy']+[cache/f'fold_{i}'/'complete.json' for i in range(1,len(self.folds)+1)])}
        write_json(marker,result)
        self.log('cv_complete',config=key,seed=seed,metrics=metrics,prediction_path=str(path))
        return result

    def evaluate_configs(self,stage,configs,seeds):
        """评估一组配置并保存逐种子及均值表，重复配置由有效参数键复用训练。"""
        results,rows = [],[]
        for config in configs:
            evaluated = [self.evaluate_cv(config,seed) for seed in seeds]
            means = {metric:float(np.mean([result['metrics'][metric] for result in evaluated])) for metric in METRICS}
            result = {'config':config,'config_key':config_key(config),'mean':means,'runs':evaluated}
            results.append(result)
            for item in evaluated:
                rows.append({'stage':stage,'experiment_id':config['experiment_id'],'config_key':config_key(config),
                             'seed':item['seed'],**item['metrics'],'prediction_path':item['prediction_path']})
            # 每个配置完成后写一次，长实验中断也能看到已完成结果。
            pd.DataFrame(rows).to_excel(self.run_dir/f'{stage}_seed_metrics.xlsx',index=False)
            write_json(self.run_dir/f'{stage}_results.json',results)
        summary = pd.DataFrame(rows).groupby(['experiment_id','config_key'])[METRICS].agg(['mean','std'])
        summary.to_excel(self.run_dir/f'{stage}_mean_std.xlsx')
        return results

    def freeze_config(self,config,seeds,calibrate=False):
        """用已完成OOF结果冻结全量训练轮数和可选阈值，保存完整配置及验证证据。"""
        results = [self.evaluate_cv(config,seed) for seed in seeds]
        epochs = [fold['best_epoch'] for result in results for fold in result['folds']]
        frozen = {**config,'final_epochs':int(math.ceil(float(np.median(epochs)))),
                  'decision_threshold':.5,'frozen':True,'protocol':PROTOCOL_VERSION,
                  'validation_seeds':list(seeds),'run_id':self.run_id,
                  'data_fingerprint':self.identity['data'],'config_key':config_key(config)}
        if calibrate:
            labels = np.tile(self.train.labels,len(results))
            probs = np.concatenate([np.load(item['probabilities_path'],allow_pickle=False) for item in results])
            frozen['calibrated_threshold'] = float(find_best_threshold(labels,probs))
        write_json(self.run_dir/'frozen'/f'{config_key(config)}.json',frozen)
        return frozen

    def evaluate_final(self,config,seed):
        """只接受本次运行的冻结配置；全量训练后一次测试，按窗口导出原目录预测。"""
        freeze_path = self.run_dir/'frozen'/f'{config_key(config)}.json'
        if not config.get('frozen') or not freeze_path.exists() or read_json(freeze_path) != config:
            raise ValueError('最终测试必须使用本次运行已保存的冻结配置')
        key = config_key(config)
        folder = self.run_dir/'final'/key/f'seed_{seed}'
        marker = folder/'complete.json'
        if marker.exists():
            saved = read_json(marker)
            verify_artifacts(saved['artifacts'],allow_missing=saved['prediction_path'])
            if not Path(saved['prediction_path']).exists():
                saved = self.restore_export(saved,self.test,self.sources['test'],'Independent_Test')
                write_json(marker,saved)
            return saved
        model,scaler,info = self.fit(config,self.train,None,seed,0,folder,epochs=config['final_epochs'])
        start = time.perf_counter()
        probabilities = predict(model,transform_windows(self.test,scaler,config['feature_mode']),config['batch_size'],self.device)
        if self.device.type=='cuda':
            torch.cuda.synchronize()
        info['inference_seconds'] = time.perf_counter()-start
        records = pd.DataFrame({'sSowsNo_split':self.test.metadata.sSowsNo_split,'y_prob':probabilities})
        path = export_predictions(self.test,records,self.sources['test'],'FINAL_'+key,seed,self.run_id,config['model_name'],
                                  threshold=.5,calibrated_threshold=config.get('calibrated_threshold'),split_index=self.split_index)
        np.save(folder/'probabilities.npy',probabilities,allow_pickle=False)
        result = {'config':config,'config_key':key,'seed':int(seed),'metrics':binary_metrics(self.test.labels,probabilities),
                  'prediction_path':str(path),'model_dir':str(folder),'probabilities_path':str(folder/'probabilities.npy'),**info,
                  'artifacts':artifact_hashes([folder/name for name in MODEL_ARTIFACTS+['probabilities.npy']]+[path])}
        if 'calibrated_threshold' in config:
            result['calibrated_metrics'] = binary_metrics(self.test.labels,probabilities,config['calibrated_threshold'])
        write_json(marker,result)
        self.log('final_complete',config=key,seed=seed,metrics=result['metrics'],prediction_path=str(path))
        return result

    def bayesian_search(self,cleaner,warm_start,n_calls=60):
        """执行确定性的离散贝叶斯搜索，仅反馈OOF F1；恢复时重放优化器并复用已完成模型。"""
        from skopt import Optimizer
        from skopt.space import Categorical
        keys = list(SEARCH_SPACE)
        optimizer = Optimizer([Categorical(list(range(len(SEARCH_SPACE[key]))),name=key) for key in keys],
                              base_estimator='GP',n_initial_points=12,random_state=766)
        write_json(self.run_dir/'E5_search_space.json',SEARCH_SPACE)
        visited,results,trace = set(),[],[]
        rng = random.Random(766)
        for iteration in range(n_calls):
            proposed = tuple(optimizer.ask())
            point = tuple(SEARCH_SPACE[key].index(warm_start[key]) for key in keys) if iteration==0 else proposed
            while point in visited:
                point = tuple(rng.randrange(len(SEARCH_SPACE[key])) for key in keys)
            visited.add(point)
            parameters = {key:SEARCH_SPACE[key][index] for key,index in zip(keys,point)}
            config = named_config(f'E5_BO_{iteration+1:03}',tomek_mode=cleaner,**parameters)
            result = self.evaluate_configs('E5_current',[config],[766])[0]
            optimizer.tell(list(point),-result['mean']['F1-Score'])
            results.append(result)
            trace.append({'iteration':iteration+1,'config_key':config_key(config),**result['mean']})
            pd.DataFrame(trace).to_excel(self.run_dir/'E5_search_trace.xlsx',index=False)
            write_json(self.run_dir/'E5_results.json',results)
        return results

    def run(self,through='E8'):
        """按依赖顺序执行E0—E8，写运行状态；出错保留缓存以便显式resume。"""
        try:
            write_json(self.run_dir/'status.json',{'status':'running','through':through,'profile':self.profile})
            if self.profile=='audit' or through=='E0':
                write_json(self.run_dir/'status.json',{'status':'complete','through':'E0'})
                return self.run_dir
            if self.profile=='smoke':
                configs = [stage_configs('E1')[0],stage_configs('E1')[-1]]
                self.evaluate_configs('smoke',configs,[766])
                for config in configs:
                    self.evaluate_final(self.freeze_config(config,[766]),766)
                self.write_final_summary()
                from experiment_reports import build_reports
                build_reports(self.run_dir)
                write_json(self.run_dir/'status.json',{'status':'complete','through':'smoke','not_for_paper':True})
                return self.run_dir
            limit = int(through[1:])
            first = self.evaluate_configs('E1',stage_configs('E1'),self.train_seeds)
            if limit < 2:
                return self.finish_stage(through)
            second = self.evaluate_configs('E2',stage_configs('E2'),self.train_seeds)
            custom = next(item for item in first if item['config']['name']=='E1_temp_rate_AST')
            standard = next(item for item in second if item['config']['name']=='E2_standard_AST')
            cleaner = rank_results([custom,standard])[0]['config']['tomek_mode']
            write_json(self.run_dir/'selected_cleaner.json',{'tomek_mode':cleaner,'selection':'mean OOF F1, AUC, Recall; tie custom'})
            if limit < 3:
                return self.finish_stage(through)
            third = self.evaluate_configs('E3',stage_configs('E3',cleaner),[766])
            if limit < 4:
                return self.finish_stage(through)
            fourth = self.evaluate_configs('E4',stage_configs('E4',cleaner,rank_results(third)[0]['config']),[766])
            if limit < 5:
                return self.finish_stage(through)
            fifth = self.bayesian_search(cleaner,rank_results(fourth)[0]['config'])
            if limit < 6:
                return self.finish_stage(through)
            sixth = self.evaluate_configs('E6',[item['config'] for item in rank_results(fifth)[:3]],self.train_seeds)
            best = self.freeze_config(rank_results(sixth)[0]['config'],self.train_seeds,calibrate=True)
            write_json(self.run_dir/'best_bilstm_config.json',best)
            if limit < 7:
                return self.finish_stage(through)
            seventh = self.evaluate_configs('E7',stage_configs('E7',cleaner),self.train_seeds)
            frozen = [self.freeze_config(item['config'],self.train_seeds) for item in seventh]
            # 若调优最优正好等于参考BiLSTM，保留校准后的同一份冻结配置并仅训练一次。
            if any(config_key(item)==config_key(best) for item in frozen):
                best['roles'] = ['fixed','tuned']
            frozen = [item for item in frozen if config_key(item)!=config_key(best)]
            write_json(self.run_dir/'frozen'/f'{config_key(best)}.json',best)
            frozen.append(best)
            write_json(self.run_dir/'final_configs.json',frozen)
            if limit < 8:
                return self.finish_stage(through)
            for config in frozen:
                for seed in self.final_seeds:
                    self.evaluate_final(config,seed)
                    self.write_final_summary()
            self.finish_stage(through)
            from experiment_reports import build_reports
            build_reports(self.run_dir)
            return self.run_dir
        except BaseException as exc:
            write_json(self.run_dir/'status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                                                  'error':str(exc),'type':type(exc).__name__})
            self.log('run_stopped',error=str(exc),type=type(exc).__name__)
            raise

    def finish_stage(self,stage):
        """记录已执行到指定阶段，返回运行目录；不将尚未请求执行的后续阶段标作完成。"""
        write_json(self.run_dir/'status.json',{'status':'complete','through':stage})
        self.log('stage_complete',stage=stage)
        return self.run_dir

    def write_final_summary(self):
        """从已完成最终模型标记重建指标与路径清单，按种子统计样本标准差。"""
        rows,calibrated = [],[]
        for marker in sorted((self.run_dir/'final').glob('*/seed_*/complete.json')):
            result = read_json(marker)
            row = {'config_key':result['config_key'],'model_name':result['config']['model_name'],
                   'name':result['config']['name'],'seed':result['seed'],'prediction_path':result['prediction_path'],
                   'model_dir':result['model_dir'],'parameters':result['parameters'],
                   'train_seconds':result['train_seconds'],'inference_seconds':result['inference_seconds']}
            rows.append({**row,**result['metrics']})
            if 'calibrated_metrics' in result:
                calibrated.append({**row,**result['calibrated_metrics']})
        if rows:
            frame = pd.DataFrame(rows)
            frame.to_excel(self.run_dir/'final_seed_metrics.xlsx',index=False)
            frame.groupby(['config_key','name'])[METRICS].agg(['mean','std']).to_excel(self.run_dir/'final_mean_std.xlsx')
        if calibrated:
            pd.DataFrame(calibrated).to_excel(self.run_dir/'final_calibrated_metrics.xlsx',index=False)


def cli(argv=None):
    """命令行入口；默认执行完整方案，audit仅审计，smoke使用两轮并隔离结果缓存。"""
    from sow_estrus_LSTM_Info import experimentRecord_data_path,result_save_path
    parser = argparse.ArgumentParser(description='按原始母猪分组的48小时发情实验；预测保存到原数据目录')
    parser.add_argument('--path-index',type=int,default=1)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--result-root',type=Path,default=Path(result_save_path)/'all_experiments')
    parser.add_argument('--profile',choices=['audit','smoke','full'],default='full')
    parser.add_argument('--through',choices=[f'E{i}' for i in range(9)],default='E8')
    parser.add_argument('--epochs',type=int,help='正式默认100轮，smoke默认2轮；写入缓存身份')
    parser.add_argument('--resume',type=Path,help='恢复既有运行目录，要求数据、代码和训练参数完全相同')
    args = parser.parse_args(argv)
    data_dir = args.data_dir or Path(experimentRecord_data_path)/'cross_validation_dataset'/str(args.path_index)
    runner = ExperimentRunner(data_dir,args.result_root,profile=args.profile,split_index=args.path_index,
                              epochs=args.epochs if args.epochs is not None else (2 if args.profile=='smoke' else 100),run_dir=args.resume)
    return runner.run(args.through)


if __name__ == '__main__':
    cli()
