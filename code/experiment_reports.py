"""从已保存的全部种子结果生成实验表图，不重新选模或挑选测试最佳种子。"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def load_json(path):
    """读取UTF-8实验记录，图表仅使用已保存结果。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_figure(fig,path):
    """将图导出为300dpi PNG及矢量SVG，然后释放绘图对象。"""
    fig.tight_layout()
    fig.savefig(path,dpi=300,bbox_inches='tight')
    fig.savefig(Path(path).with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def configuration_label(config):
    """将配置名转为明确区分统一设置和专项调优的图例标签。"""
    name = config['name']
    if set(config.get('roles',[]))=={'fixed','tuned'}:
        return 'BiLSTM (fixed = tuned)'
    if name.startswith('E7_'):
        return name[3:]+' (fixed)'
    if name.startswith('E5_'):
        return 'BiLSTM (tuned)'
    return name


def plot_validation(run_dir,figures):
    """绘制验证阶段的特征消融、清理对照、结构/倍率趋势和贝叶斯搜索轨迹。"""
    e1_path = run_dir/'E1_results.json'
    if e1_path.exists():
        results = load_json(e1_path)
        fig,ax = plt.subplots(figsize=(10,4))
        names = ['Baseline','A','S','T','AS','AT','ST','AST']
        grid = np.asarray([[next(item['mean']['F1-Score'] for item in results
                                 if item['config']['name']==f'E1_{mode}_{name}') for name in names]
                           for mode in ['temp_only','temp_rate']])
        panel = ax.imshow(grid,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_xticks(range(8),names)
        ax.set_yticks([0,1],['Temperature','Temperature + difference'])
        for row in range(2):
            for col in range(8):
                ax.text(col,row,f'{grid[row,col]:.3f}',ha='center',va='center',color='white' if grid[row,col]>.6 else 'black')
        fig.colorbar(panel,ax=ax,label='Mean OOF F1 (3 seeds)')
        save_figure(fig,figures/'feature_sampling_ablation.png')
    e2_path = run_dir/'E2_results.json'
    if e1_path.exists() and e2_path.exists():
        first,second = load_json(e1_path),load_json(e2_path)
        names = ['T','AT','ST','AST']
        fig,ax = plt.subplots(figsize=(7,4))
        for offset,items,prefix,label in [(-.18,first,'E1_temp_rate_','Custom'),(.18,second,'E2_standard_','Standard')]:
            values = [next(item['mean']['F1-Score'] for item in items if item['config']['name']==prefix+name) for name in names]
            ax.bar(np.arange(4)+offset,values,width=.36,label=label)
        ax.set_xticks(range(4),names)
        ax.set_ylabel('Mean OOF F1 (3 seeds)')
        ax.legend()
        save_figure(fig,figures/'cleaning_comparison.png')
    for stage,column,title in [('E3','hidden_sizes','Hidden layers'),('E4','smote_amount','Additional samples per input (%)')]:
        path = run_dir/f'{stage}_results.json'
        if not path.exists():
            continue
        results = load_json(path)
        labels = [str(item['config'][column]) for item in results]
        fig,ax = plt.subplots(figsize=(9,4))
        for metric in ['F1-Score','Recall','Precision','MCC']:
            ax.plot(range(len(results)),[item['mean'][metric] for item in results],marker='o',label=metric)
        ax.set_xticks(range(len(results)),labels,rotation=20 if stage=='E3' else 0)
        ax.set_xlabel(title)
        ax.set_ylabel('OOF metric (screening seed 766)')
        ax.legend()
        save_figure(fig,figures/f'{stage}_sensitivity.png')
        pd.DataFrame([{'configuration':item['config']['name'],column:str(item['config'][column]),
                       'parameters':item['runs'][0]['folds'][0]['parameters'],**item['mean']}
                      for item in results]).to_excel(run_dir/f'{stage}_performance_parameters.xlsx',index=False)
    trace_path = run_dir/'E5_search_trace.xlsx'
    if trace_path.exists():
        trace = pd.read_excel(trace_path)
        fig,ax = plt.subplots(figsize=(8,4))
        ax.scatter(trace.iteration,trace['F1-Score'],alpha=.6,label='Candidate')
        ax.plot(trace.iteration,trace['F1-Score'].cummax(),label='Best so far')
        ax.set(xlabel='Bayesian iteration',ylabel='OOF F1')
        ax.legend()
        save_figure(fig,figures/'bayesian_search.png')


def plot_final(run_dir,figures):
    """展示所有最终种子的ROC/PR、F1分布和汇总混淆矩阵，不选择单个最佳种子。"""
    grouped = {}
    for marker in sorted((run_dir/'final').glob('*/seed_*/complete.json')):
        result = load_json(marker)
        grouped.setdefault(result['config_key'],[]).append(result)
    if not grouped:
        return
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    colors = plt.get_cmap('tab10').colors
    for index,results in enumerate(grouped.values()):
        color = colors[index%len(colors)]
        label = configuration_label(results[0]['config'])
        for seed_index,result in enumerate(results):
            frame = pd.read_excel(result['prediction_path'])
            fpr,tpr,_ = roc_curve(frame.isEstrus,frame.y_prob)
            precision,recall,_ = precision_recall_curve(frame.isEstrus,frame.y_prob)
            legend = f'{label}; n={len(results)}' if seed_index==0 else None
            axes[0].plot(fpr,tpr,color=color,alpha=.45,label=legend)
            axes[1].plot(recall,precision,color=color,alpha=.45,label=legend)
    axes[0].plot([0,1],[0,1],linestyle='--',color='grey')
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC: all seeds')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='PR: all seeds')
    for ax in axes:
        ax.legend(fontsize=7)
    save_figure(fig,figures/'final_roc_pr.png')
    columns = min(3,len(grouped))
    rows = math.ceil(len(grouped)/columns)
    fig,axes = plt.subplots(rows,columns,figsize=(4*columns,3.6*rows),squeeze=False)
    for ax,results in zip(axes.flat,grouped.values()):
        counts = np.zeros((2,2),dtype=int)
        for result in results:
            frame = pd.read_excel(result['prediction_path'])
            counts += confusion_matrix(frame.isEstrus,frame.y_pred,labels=[0,1])
        ax.imshow(counts,cmap='Blues')
        for row in range(2):
            for col in range(2):
                ax.text(col,row,str(counts[row,col]),ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='True',
               title=configuration_label(results[0]['config'])+f'\nSum over {len(results)} seeds')
    for ax in list(axes.flat)[len(grouped):]:
        ax.set_visible(False)
    save_figure(fig,figures/'final_confusion.png')
    fig,ax = plt.subplots(figsize=(10,4))
    values = [[item['metrics']['F1-Score'] for item in results] for results in grouped.values()]
    labels = [configuration_label(results[0]['config']) for results in grouped.values()]
    ax.boxplot(values,tick_labels=labels)
    for index,scores in enumerate(values,1):
        ax.scatter(np.full(len(scores),index),scores,alpha=.7)
    ax.set_ylabel('Independent test F1, threshold 0.5')
    save_figure(fig,figures/'final_seed_f1.png')


def plot_loss_histories(run_dir,figures):
    """按最终配置展示全部CV训练/验证与最终训练损失曲线，不挑选最好的一折或种子。"""
    completed = {}
    for marker in sorted((run_dir/'final').glob('*/seed_*/complete.json')):
        result = load_json(marker)
        completed.setdefault(result['config_key'],result['config'])
    if not completed:
        return
    fig,axes = plt.subplots(len(completed),2,figsize=(12,3*len(completed)),squeeze=False)
    for row,(key,config) in enumerate(completed.items()):
        first = True
        for path in sorted((run_dir/'cv'/key).glob('seed_*/fold_*/history.csv')):
            history = pd.read_csv(path)
            axes[row,0].plot(history.epoch,history.train_loss,color='tab:blue',alpha=.3,label='Train' if first else None)
            axes[row,0].plot(history.epoch,history.val_loss,color='tab:orange',alpha=.3,label='Validation' if first else None)
            first = False
        for path in sorted((run_dir/'final'/key).glob('seed_*/history.csv')):
            history = pd.read_csv(path)
            axes[row,1].plot(history.epoch,history.train_loss,alpha=.6,label=path.parent.name)
        for col,title in [(0,'CV: all folds/seeds'),(1,'Final: all seeds')]:
            axes[row,col].set(xlabel='Epoch',ylabel='BCE loss',title=configuration_label(config)+' / '+title)
            axes[row,col].legend(fontsize=6)
    save_figure(fig,figures/'training_validation_loss.png')


def build_reports(run_dir):
    """重建静态图表及简要Markdown说明；不更改预测、模型或冻结配置。"""
    run_dir = Path(run_dir)
    figures = run_dir/'figures'
    figures.mkdir(exist_ok=True)
    plot_validation(run_dir,figures)
    plot_final(run_dir,figures)
    plot_loss_histories(run_dir,figures)
    lines = ['# 实验结果索引','',
             '主指标使用固定阈值0.5。验证指标按每种子完整OOF计算；最终结果按全部种子汇总，标准差使用ddof=1。',
             '混淆矩阵累加各次预测，计数包含重复测试窗口，不能解释为独立动物数。',
             '动态阈值仅由训练/验证OOF选择，补充结果单独保存。','',
             '- 数据及折清单：dataset_counts.xlsx、fold_manifest.xlsx',
             '- 配置与可复现信息：manifest.json、frozen/、best_bilstm_config.json',
             '- 逐种子及汇总：final_seed_metrics.xlsx、final_mean_std.xlsx',
             '- 预测：位于原输入文件同目录；具体路径见逐种子指标表或各模型complete.json。','',
             '## 图表','']
    lines += [f'- [{path.stem}](figures/{path.name})' for path in sorted(figures.glob('*.png'))]
    (run_dir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
