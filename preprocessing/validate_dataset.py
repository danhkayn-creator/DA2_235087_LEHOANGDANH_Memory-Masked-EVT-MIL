# Nếu thiếu thư viện, bỏ dấu # ở dòng dưới và chạy:
# %pip install numpy pandas scikit-learn matplotlib
from pathlib import Path
import re, json
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Colab: mount Drive nếu cần, rồi chỉnh ROOT.
# from google.colab import drive
# drive.mount('/content/drive')
import os
ROOT = Path(os.environ.get('VAD_ROOT', Path(__file__).resolve().parent.parent)).resolve()
FEATURES = ROOT/'data/processed/clip_features'
ANNOTATIONS = ROOT/'data/annotations'
TRAIN_LIST = ANNOTATIONS/'ucf_CLIP_rgb.csv'
TEST_LIST = ANNOTATIONS/'ucf_CLIP_rgbtest.csv'
SEED = 42
VAL_RATIO = 0.10
DIM = 512
EXPECTED_TRAIN, EXPECTED_TEST = 1610, 290




def validate():
    files = sorted(FEATURES.rglob('*.npy'))
    assert files, 'Không tìm thấy NPY.'
    print('Số file NPY:', len(files))
    for p in files[:10]: print(p.relative_to(FEATURES))
    for p in [TRAIN_LIST, TEST_LIST]:
        assert p.is_file(), f'Thiếu danh sách chính thức: {p}'
        print('\n', p.name)
        print('\n'.join(p.read_text(encoding='utf-8-sig').splitlines()[:5]))

    # Chỉ nhận dạng ID video UCF-Crime; hậu tố crop/segment không tạo video mới.
    CATEGORIES = ['Normal','Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']
    pattern = re.compile(r'(Normal(?:_?Videos?)?(?:_?For_?Anomaly)?|Road_?Accidents?|Abuse|Arrest|Arson|Assault|Burglary|Explosion|Fighting|Robbery|Shooting|Shoplifting|Stealing|Vandalism)[_-]?(\d+)', re.I)
    def video_info(path):
        name = str(path).replace(chr(92), '/').rsplit('/',1)[-1]
        m = pattern.search(name)
        if not m: raise ValueError(f'Không nhận dạng được video ID: {name}')
        prefix, number = m.groups()
        category = 'Normal' if prefix.lower().startswith('normal') else ('RoadAccidents' if prefix.lower().startswith('road') else next(c for c in CATEGORIES if c.lower()==prefix.lower()))
        return category + str(int(number)).zfill(3), category

    by_name = defaultdict(list)
    for p in files: by_name[p.name.casefold()].append(p)

    def load_list(path, split):
        table = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
        rows=[]; errors=[]
        for i,row in table.iterrows():
            candidates=[v.strip() for v in row if str(v).strip().lower().endswith('.npy')]
            if not candidates:
                if i==0 and any(str(v).strip().lower() in ('path','feature_path','video_path','filename','file','label','index') for v in row): continue
                errors.append(f'Dòng {i+1}: không có đường dẫn NPY'); continue
            ref=candidates[0].replace(chr(92),'/'); name=ref.rsplit('/',1)[-1]
            matches=by_name.get(name.casefold(),[])
            if len(matches)>1:
                matches=[p for p in matches if ref.casefold().endswith('/'+p.relative_to(FEATURES).as_posix().casefold())]
            if len(matches)!=1:
                errors.append(f'Dòng {i+1}: {name}, số file khớp={len(matches)}'); continue
            try: vid, cat=video_info(name)
            except ValueError as e: errors.append(str(e)); continue
            rows.append(dict(video_id=vid,feature_path=matches[0].relative_to(FEATURES).as_posix(),category=cat,label=int(cat!='Normal'),split=split))
        if errors: raise ValueError(f'{path.name}: {len(errors)} lỗi.\n'+'\n'.join(errors[:20]))
        assert rows, f'Danh sách rỗng: {path}'
        return pd.DataFrame(rows).drop_duplicates(subset=['feature_path'])
    train_all=load_list(TRAIN_LIST,'train')
    test=load_list(TEST_LIST,'test')
    assert not set(train_all.video_id)&set(test.video_id), 'Video trùng giữa Train và Test.'
    assert not set(train_all.feature_path)&set(test.feature_path), 'Feature trùng giữa Train và Test.'
    print('Train:', train_all.video_id.nunique(), 'Test:', test.video_id.nunique())
    assert train_all.video_id.nunique()==EXPECTED_TRAIN and test.video_id.nunique()==EXPECTED_TEST, 'Số video khác chuẩn UCF-Crime. Kiểm tra ID, CSV, dữ liệu trước khi tiếp tục; chỉ chỉnh EXPECTED nếu có lý do rõ ràng.'

    records=[]; errors=[]
    for rel in sorted(set(train_all.feature_path)|set(test.feature_path)):
        try:
            a=np.load(FEATURES/rel, mmap_mode='r', allow_pickle=False)
            if a.ndim!=2 or a.shape[0]==0 or a.shape[1]!=DIM: raise ValueError(f'shape={a.shape}')
            if not np.issubdtype(a.dtype,np.floating): raise ValueError(f'dtype={a.dtype}')
            if not np.isfinite(a).all(): raise ValueError('NaN/Inf')
            records.append(dict(feature_path=rel,timesteps=int(a.shape[0]),dtype=str(a.dtype)))
        except Exception as e: errors.append(f'{rel}: {e}')
    print('Hợp lệ:',len(records),'Lỗi:',len(errors))
    assert not errors, '\n'.join(errors[:20])
    meta=pd.DataFrame(records)
    print(meta.timesteps.describe())
    unused=set(p.relative_to(FEATURES).as_posix() for p in files)-set(meta.feature_path)
    print('File NPY không được tham chiếu:',len(unused))

    return train_all, test, meta, unused

if __name__ == '__main__':
    validate()
