# label_mapper.py
import pandas as pd

def merge_real_labels(emg_samples, metadata_path, subject_col='subject_id', label_col='diagnosis_level'):

   
    try:
        metadata_df = pd.read_csv(metadata_path)
        
        metadata_df[subject_col] = metadata_df[subject_col].astype(str)
        
        label_map = dict(zip(metadata_df[subject_col], metadata_df[label_col]))
    except Exception as e:
        return []

    labeled_samples = []
    missing_ids = set()

    for sample in emg_samples:
        sid = str(sample['subject_id'])
        if sid in label_map:
            sample['label'] = int(label_map[sid])
            labeled_samples.append(sample)
        else:
            missing_ids.add(sid)

    
    if missing_ids:
        pass

    return labeled_samples