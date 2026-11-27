from validate_dataset import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATEGORIES = ['Normal','Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']

def statistics():
    frames=[]
    for name in ('train','val','test'):
        df=pd.read_csv(ROOT/'data/splits'/f'{name}.csv')
        if not (df.split==name).all(): raise ValueError(f'Split mismatch: {name}')
        frames.append(df)
    all_rows=pd.concat(frames,ignore_index=True)
    if not all_rows.feature_path.is_unique: raise ValueError('Duplicate feature paths')
    video_rows=all_rows[['video_id','category','label','split']].drop_duplicates()
    if not video_rows.video_id.is_unique: raise ValueError('Video leakage or inconsistent labels')
    unused=set(p.relative_to(FEATURES).as_posix() for p in FEATURES.rglob('*.npy'))-set(all_rows.feature_path)
    # 4. Xuất CSV, JSON và biểu đồ thực tế
    SPLITS=ROOT/'data/splits'; STATS=ROOT/'data/statistics'
    SPLITS.mkdir(parents=True,exist_ok=True); STATS.mkdir(parents=True,exist_ok=True)
    counts=pd.crosstab(video_rows.category,video_rows.split).reindex(index=CATEGORIES,columns=['train','val','test'],fill_value=0)
    counts['total']=counts.sum(axis=1)
    counts.to_csv(STATS/'class_distribution.csv')
    split_counts=video_rows.split.value_counts().reindex(['train','val','test'])
    summary=dict(dataset='UCF-Crime',feature_source='VadCLIP',feature_dim=DIM,total_videos=int(len(video_rows)),num_classes=int(video_rows.category.nunique()),feature_files=int(len(all_rows)),unused_npy_files=len(unused),seed=SEED,val_ratio=VAL_RATIO,splits={k:int(v) for k,v in split_counts.items()},binary_distribution={str(k):int(v) for k,v in video_rows.label.value_counts().items()},augmentation='none',balancing='balanced video sampling planned at training time',train_list=TRAIN_LIST.name,test_list=TEST_LIST.name)
    (STATS/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    ax=counts[['train','val','test']].plot.bar(stacked=True,figsize=(12,5)); ax.set_ylabel('Videos'); ax.set_xlabel('Category'); ax.set_title('UCF-Crime class distribution'); plt.tight_layout(); plt.savefig(STATS/'class_distribution.png',dpi=160); plt.close()
    ax=split_counts.plot.bar(figsize=(6,4),rot=0); ax.set_ylabel('Videos'); ax.set_title('Train / Validation / Test'); ax.bar_label(ax.containers[0]); plt.tight_layout(); plt.savefig(STATS/'split_distribution.png',dpi=160); plt.close()
    print(json.dumps(summary,indent=2)); print('Output:',SPLITS,STATS)


if __name__ == '__main__':
    statistics()
