from validate_dataset import *

def split_dataset():
    train_all, test, meta, unused = validate()
    # 3. Chia validation theo ID video, phân tầng theo category.
    unique=train_all[['video_id','category','label']].drop_duplicates()
    assert unique.video_id.is_unique
    tr_ids, va_ids=train_test_split(unique.video_id, test_size=VAL_RATIO, random_state=SEED, stratify=unique.category)
    train=train_all[train_all.video_id.isin(tr_ids)].copy()
    val=train_all[train_all.video_id.isin(va_ids)].copy(); val['split']='val'
    assert not set(train.video_id)&set(val.video_id)
    assert not set(train.video_id)&set(test.video_id)
    assert not set(val.video_id)&set(test.video_id)
    assert len(train)+len(val)==len(train_all)
    all_rows=pd.concat([train,val,test],ignore_index=True).merge(meta,on='feature_path',validate='one_to_one')
    video_rows=all_rows[['video_id','category','label','split']].drop_duplicates()
    assert video_rows.video_id.is_unique
    print(pd.crosstab(video_rows.category,video_rows.split))

    splits = ROOT/'data/splits'
    splits.mkdir(parents=True, exist_ok=True)
    for name in ('train','val','test'):
        all_rows[all_rows.split==name].sort_values(['video_id','feature_path']).to_csv(splits/f'{name}.csv',index=False)
    print('Saved:', splits)
    return all_rows

if __name__ == '__main__':
    split_dataset()
