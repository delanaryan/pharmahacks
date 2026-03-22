#main.py
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold

from denoise import denoise_eeg
from proc import (compute_rbp, compute_scc, compute_hjorth,
                  compute_entropy, compute_plv, compute_plv_per_band)


# ============================================================
# LOADING
# ============================================================

def load_data_from_folders(base_path='training'):
    """
    Loads and denoises all EEG recordings from AD and CN directories.
    Subject ID is derived from filename (e.g. '3.npy' -> '3').
    """
    data_list, labels, subject_ids = [], [], []
    class_mapping = {'AD': 1, 'CN': 0}

    for folder_name, label_value in class_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        print(f"\nLoading {len(files)} subjects from {folder_name}...")

        for i, file_name in enumerate(files):
            sub_id = file_name.replace('.npy', '')
            print(f"  [{i+1}/{len(files)}] Subject {sub_id}...", end=' ')
            data = np.load(os.path.join(folder_path, file_name))
            data_denoised = denoise_eeg(data)
            print(f"done. Shape: {data_denoised.shape}")
            data_list.append(data_denoised)
            labels.append(label_value)
            subject_ids.append(sub_id)

    print(f"\nLoaded {len(data_list)} subjects "
          f"({labels.count(1)} AD, {labels.count(0)} CN)")
    return data_list, labels, subject_ids


# ============================================================
# FEATURE CACHING — extract everything once
# ============================================================

def build_feature_cache(data_list, labels, subject_ids, sfreq=128):
    """
    Extracts ALL features from every epoch exactly once and stores
    them in a cache. main() and experiments.py both use this cache
    to avoid redundant computation.

    Cache structure:
        epoch_cache: list of dicts, one per epoch, keys are feature names
        all_labels:  list of int labels per epoch
        all_groups:  list of subject ID strings per epoch

    Feature shapes per epoch (after mean over time axis):
        rbp:      (95,)  — 5 bands x 19 channels
        scc:      (95,)
        hjorth:   (57,)  — 3 params x 19 channels
        entropy:  (19,)  — 1 x 19 channels
        plv:      (19,)
        plv_band: (95,)  — 5 bands x 19 channels
    """
    window_size = 30 * sfreq
    step_size   = 15 * sfreq
    total = len(data_list)

    epoch_cache, all_labels, all_groups = [], [], []

    for subj_idx, (data, label, sub_id) in enumerate(
            zip(data_list, labels, subject_ids)):

        n_epochs = len(range(0, data.shape[1] - window_size + 1, step_size))
        print(f"  [{subj_idx+1}/{total}] Subject {sub_id} "
              f"({'AD' if label==1 else 'CN'}) "
              f"— {n_epochs} epochs...", end=' ')

        for start in range(0, data.shape[1] - window_size + 1, step_size):
            epoch = data[:, start:start + window_size]

            # Extract all features — each called once per epoch
            rbp      = compute_rbp(epoch, sfreq)
            scc      = compute_scc(epoch, sfreq)
            hjorth   = compute_hjorth(epoch)
            entropy  = compute_entropy(epoch)
            plv      = compute_plv(epoch)
            plv_band = compute_plv_per_band(epoch, sfreq)

            # Average over time dimension and flatten
            epoch_cache.append({
                'rbp':      np.mean(rbp,      axis=0).flatten(),
                'scc':      np.mean(scc,      axis=0).flatten(),
                'hjorth':   np.mean(hjorth,   axis=0).flatten(),
                'entropy':  np.mean(entropy,  axis=0).flatten(),
                'plv':      np.mean(plv,      axis=0).flatten(),
                'plv_band': np.mean(plv_band, axis=0).flatten(),
            })
            all_labels.append(label)
            all_groups.append(sub_id)

        print("done.")

    print(f"\nCached {len(epoch_cache)} epochs total.")
    return epoch_cache, np.array(all_labels), np.array(all_groups)


def build_X(epoch_cache, feature_combo):
    """
    Builds feature matrix from cache for a given feature combination.
    No recomputation — just concatenates the requested feature vectors.

    Args:
        epoch_cache: list of dicts from build_feature_cache
        feature_combo: list of feature names e.g. ['rbp', 'scc']
    Returns:
        X: (n_epochs, n_features)
    """
    return np.array([
        np.hstack([epoch[f] for f in feature_combo])
        for epoch in epoch_cache
    ])


# ============================================================
# TRAINING + EVALUATION
# ============================================================

def run_experiment(X, y, groups, feature_combo, svm_params):
    """
    Runs one full GroupKFold CV experiment.
    Subject-level majority voting for final predictions.

    Args:
        X: (n_epochs, n_features)
        y: (n_epochs,) labels
        groups: (n_epochs,) subject IDs
        feature_combo: list of feature names (for logging)
        svm_params: dict e.g. {'kernel': 'rbf', 'C': 1.0}
    Returns:
        dict with all metrics
    """
    gkf = GroupKFold(n_splits=5)
    subj_actuals, subj_preds = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = SVC(class_weight='balanced', **svm_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        for sid in np.unique(groups_test):
            indices = np.where(groups_test == sid)[0]
            votes = preds[indices]
            subj_preds.append(np.bincount(votes).argmax())
            subj_actuals.append(y_test[indices[0]])

    acc  = accuracy_score(subj_actuals, subj_preds)
    prec = precision_score(subj_actuals, subj_preds, zero_division=0)
    rec  = recall_score(subj_actuals, subj_preds, zero_division=0)
    f1   = f1_score(subj_actuals, subj_preds, zero_division=0)

    return {
        'features':   '+'.join(feature_combo),
        'kernel':     svm_params['kernel'],
        'C':          svm_params['C'],
        'accuracy':   acc,
        'precision':  prec,
        'recall':     rec,
        'f1':         f1,
        'n_features': X.shape[1],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    sfreq = 128

    print("=" * 60)
    print("STEP 1: Loading and Denoising")
    print("=" * 60)
    data_list, labels_list, subject_ids_list = load_data_from_folders('training')

    print("\n" + "=" * 60)
    print("STEP 2: Extracting and caching all features")
    print("=" * 60)
    epoch_cache, y, groups = build_feature_cache(
        data_list, labels_list, subject_ids_list, sfreq)

    # Default experiment: rbp + scc, rbf kernel, C=1
    print("\n" + "=" * 60)
    print("STEP 3: Running default experiment (rbp + scc)")
    print("=" * 60)
    feature_combo = ['rbp', 'scc']
    svm_params    = {'kernel': 'rbf', 'C': 1.0}

    X = build_X(epoch_cache, feature_combo)
    print(f"Feature matrix: {X.shape}")

    result = run_experiment(X, y, groups, feature_combo, svm_params)

    print(f"\nResults:")
    print(f"  Features:  {result['features']}")
    print(f"  Kernel:    {result['kernel']}  C: {result['C']}")
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")

    return epoch_cache, y, groups


if __name__ == "__main__":
    main()