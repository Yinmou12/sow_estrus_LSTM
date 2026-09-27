"""已确认E0—E8实验的配置、预算和稳定配置标识。"""
import hashlib
import json

TRAIN_SEEDS = [766,929,208]
FINAL_SEEDS = [766,929,208,676,211,443,616,558,59,131]
BASE_CONFIG = dict(model_name='EstrusLSTM', hidden_sizes=[64,64,64,64],
                   learning_rate=5e-4, dropout_rate=.2, batch_size=32,
                   weight_decay=1e-4, bidirectional=True, use_cell_state=False,
                   A=True, S=True, T=True, smote_amount=800, tomek_mode='custom',
                   feature_mode='temp_rate', early_patience=7, lr_patience=5)
AUGMENTATIONS = [('Baseline',False,False,False),('A',True,False,False),
                 ('S',False,True,False),('T',False,False,True),
                 ('AS',True,True,False),('AT',True,False,True),
                 ('ST',False,True,True),('AST',True,True,True)]
SEARCH_SPACE = {
    'hidden_sizes':[[32,32],[64,32],[64,64],[64,64,32],[128,64,32],
                    [64,64,64,64],[128,128,64,32],[64],[64,64,64]],
    'learning_rate':[1e-4,2e-4,3e-4,5e-4,7e-4,1e-3,2e-3],
    'dropout_rate':[.1,.2,.3,.4], 'batch_size':[16,32,64],
    'weight_decay':[0.,1e-5,1e-4,5e-4,1e-3],
    'smote_amount':list(range(100,1001,100)),
}


def effective_config(config):
    """提取影响训练的参数，排除展示名；关闭的采样参数归一化以便跨阶段复用。"""
    result = {key:config.get(key,value) for key,value in BASE_CONFIG.items()}
    result['hidden_sizes'] = list(result['hidden_sizes'])
    if not result['S']:
        result['smote_amount'] = 0
    if not result['T']:
        result['tomek_mode'] = 'none'
    return result


def config_key(config):
    """返回有效配置的稳定SHA256短标识，不受实验展示名或执行顺序影响。"""
    return hashlib.sha256(json.dumps(effective_config(config),sort_keys=True).encode()).hexdigest()[:16]


def named_config(name, **overrides):
    """复制参考配置并生成含内容指纹的唯一实验标识。"""
    config = {**BASE_CONFIG, **overrides, 'name':name}
    config['experiment_id'] = name+'_'+config_key(config)[:8]
    return config


def stage_configs(stage, cleaner='custom', best_structure=None):
    """构造指定阶段完整配置列表；不执行训练或修改传入配置。"""
    if stage == 'E1':
        return [named_config(f'E1_{mode}_{name}',feature_mode=mode,A=a,S=s,T=t)
                for mode in ['temp_only','temp_rate'] for name,a,s,t in AUGMENTATIONS]
    if stage == 'E2':
        return [named_config(f'E2_standard_{name}',A=a,S=s,T=t,tomek_mode='standard')
                for name,a,s,t in AUGMENTATIONS if t]
    if stage == 'E3':
        return [named_config('E3_depth_'+'_'.join(map(str,widths)),hidden_sizes=widths,tomek_mode=cleaner)
                for widths in [[64],[64,64],[64,64,64],[64]*4,[64,64,32]]]
    if stage == 'E4':
        widths = (best_structure or BASE_CONFIG)['hidden_sizes']
        return [named_config(f'E4_smote_{amount}',hidden_sizes=widths,tomek_mode=cleaner,smote_amount=amount)
                for amount in range(100,1001,100)]
    if stage == 'E7':
        return [named_config('E7_'+label,model_name=model,bidirectional=bi,tomek_mode=cleaner)
                for label,model,bi in [('RNN','EstrusRNN',False),('LSTM','EstrusLSTM',False),
                                       ('GRU','EstrusGRU',False),('BiLSTM','EstrusLSTM',True)]]
    raise ValueError(f'阶段 {stage} 不是静态配置集合')
