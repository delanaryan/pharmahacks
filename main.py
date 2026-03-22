#main.py
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold
import argparse

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
    print(f"Looking in: {os.path.abspath(base_path)}")
    print(f"Exists: {os.path.exists(base_path)}")
    if os.path.exists(base_path):
        print(f"Contents: {os.listdir(base_path)}")
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


def load_test_data(test_path='./testing/AD vs CN'):
    """
    Loads and denoises all EEG recordings from the test directory.
    Expects flat structure: all .npy files directly in test_path.
    """
    data_list, subject_ids = [], []

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test path not found: {test_path}")

    files = sorted([f for f in os.listdir(test_path) if f.endswith('.npy')])
    print(f"\nFound {len(files)} test subjects in '{test_path}'")

    for i, file_name in enumerate(files):
        sub_id = file_name.replace('.npy', '')
        print(f"  [{i+1}/{len(files)}] Subject {sub_id}...", end=' ')
        data = np.load(os.path.join(test_path, file_name))
        data_denoised = denoise_eeg(data)
        print(f"done. Shape: {data_denoised.shape}")
        data_list.append(data_denoised)
        subject_ids.append(sub_id)

    print(f"\nLoaded {len(data_list)} test subjects total.")
    return data_list, subject_ids


# ============================================================
# FEATURE CACHING — extract everything once
# ============================================================

def build_feature_cache(data_list, labels, subject_ids, sfreq=128):
    """
    Extracts ALL features from every epoch exactly once and stores
    them in a cache. main() and experiments.py both use this cache
    to avoid redundant computation.

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

            rbp      = compute_rbp(epoch, sfreq)
            scc      = compute_scc(epoch, sfreq)
            hjorth   = compute_hjorth(epoch)
            entropy  = compute_entropy(epoch)
            plv      = compute_plv(epoch)
            plv_band = compute_plv_per_band(epoch, sfreq)

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


def build_feature_cache_unlabeled(data_list, subject_ids, sfreq=128):
    """
    Same as build_feature_cache but for unlabeled test data.
    No labels — returns cache and groups only.
    """
    window_size = 30 * sfreq
    step_size   = 15 * sfreq
    total = len(data_list)

    epoch_cache, all_groups = [], []

    for subj_idx, (data, sub_id) in enumerate(zip(data_list, subject_ids)):
        n_epochs = len(range(0, data.shape[1] - window_size + 1, step_size))
        print(f"  [{subj_idx+1}/{total}] Subject {sub_id} "
              f"— {n_epochs} epochs...", end=' ')

        for start in range(0, data.shape[1] - window_size + 1, step_size):
            epoch = data[:, start:start + window_size]

            rbp      = compute_rbp(epoch, sfreq)
            scc      = compute_scc(epoch, sfreq)
            hjorth   = compute_hjorth(epoch)
            entropy  = compute_entropy(epoch)
            plv      = compute_plv(epoch)
            plv_band = compute_plv_per_band(epoch, sfreq)

            epoch_cache.append({
                'rbp':      np.mean(rbp,      axis=0).flatten(),
                'scc':      np.mean(scc,      axis=0).flatten(),
                'hjorth':   np.mean(hjorth,   axis=0).flatten(),
                'entropy':  np.mean(entropy,  axis=0).flatten(),
                'plv':      np.mean(plv,      axis=0).flatten(),
                'plv_band': np.mean(plv_band, axis=0).flatten(),
            })
            all_groups.append(sub_id)

        print("done.")

    print(f"\nCached {len(epoch_cache)} test epochs total.")
    return epoch_cache, np.array(all_groups)


def build_X(epoch_cache, feature_combo):
    """
    Builds feature matrix from cache for a given feature combination.
    No recomputation — just concatenates the requested feature vectors.
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

    Returns:
        dict with all metrics
    """
    gkf = GroupKFold(n_splits=5)
    subj_actuals, subj_preds = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        train_subjects = len(np.unique(groups[train_idx]))
        test_subjects  = len(np.unique(groups_test))
        print(f"  Fold {fold+1}/5 — {train_subjects} train, "
              f"{test_subjects} test subjects...", end=' ')

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = SVC(class_weight='balanced', **svm_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_actuals, fold_preds = [], []
        for sid in np.unique(groups_test):
            indices = np.where(groups_test == sid)[0]
            votes = preds[indices]
            final = np.bincount(votes).argmax()
            subj_preds.append(final)
            subj_actuals.append(y_test[indices[0]])
            fold_actuals.append(y_test[indices[0]])
            fold_preds.append(final)

        print(f"Acc: {accuracy_score(fold_actuals, fold_preds):.4f} "
              f"F1: {f1_score(fold_actuals, fold_preds):.4f}")

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


def train_final_model(X, y, svm_params):
    """
    Trains a single SVM on the FULL training set.
    Used for test set predictions — no CV, no held-out data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(class_weight='balanced', **svm_params)
    model.fit(X_scaled, y)
    print(f"  Trained on {X.shape[0]} epochs, {X.shape[1]} features.")
    return model, scaler


def predict_test_set(model, scaler, test_cache, test_groups,
                     feature_combo, output_path='predictions.csv'):
    """
    Generates subject-level predictions on the test set via majority
    voting and saves to CSV.

    Output CSV columns:
        anonymized_id, label (A/C), label_numeric, n_epochs,
        ad_votes, cn_votes
    """
    X_test = build_X(test_cache, feature_combo)
    X_test_scaled = scaler.transform(X_test)
    epoch_preds = model.predict(X_test_scaled)

    results = []
    for sid in np.unique(test_groups):
        indices = np.where(test_groups == sid)[0]
        votes = epoch_preds[indices]
        final_pred = np.bincount(votes).argmax()
        label_str = 'A' if final_pred == 1 else 'C'

        results.append({
            'anonymized_id': sid,
            'label':         label_str,
            'label_numeric': final_pred,
            'n_epochs':      len(indices),
            'ad_votes':      int(np.sum(votes == 1)),
            'cn_votes':      int(np.sum(votes == 0)),
        })

        print(f"  Subject {sid:>4s}: {len(indices)} epochs → "
              f"{int(np.sum(votes==1))} AD / {int(np.sum(votes==0))} CN "
              f"→ {label_str}")

    df = pd.DataFrame(results).sort_values('anonymized_id')
    df.to_csv(output_path, index=False)

    print(f"\nPredictions saved to '{output_path}'")
    print(f"  AD predicted: {(df['label']=='A').sum()}")
    print(f"  CN predicted: {(df['label']=='C').sum()}")
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEG AD vs CN Classification')
    parser.add_argument('--features', nargs='+',
                        default=['rbp', 'scc'],
                        choices=['rbp', 'scc', 'hjorth',
                                 'entropy', 'plv', 'plv_band'],
                        help='Features to use (default: rbp scc)')
    parser.add_argument('--kernel', default='rbf',
                        choices=['rbf', 'linear'],
                        help='SVM kernel (default: rbf)')
    parser.add_argument('--C', type=float, default=10.0,
                        help='SVM regularization parameter (default: 10.0)')
    parser.add_argument('--test_path', default='testing/AD vs CN',
                        help='Path to test data directory')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output CSV path for predictions')
    args = parser.parse_args()

    sfreq = 128

    # --- Step 1: Load training data ---
    print("=" * 60)
    print("STEP 1: Loading and Denoising Training Data")
    print("=" * 60)
    data_list, labels_list, subject_ids_list = load_data_from_folders('training')

    # --- Step 2: Cache all features ---
    print("\n" + "=" * 60)
    print("STEP 2: Extracting and Caching All Features")
    print("=" * 60)
    epoch_cache, y, groups = build_feature_cache(
        data_list, labels_list, subject_ids_list, sfreq)

    # --- Step 3: CV experiment ---
    print("\n" + "=" * 60)
    print("STEP 3: Cross-Validation")
    print(f"  Features: {args.features}")
    print(f"  Kernel:   {args.kernel}  C: {args.C}")
    print("=" * 60)

    X_train = build_X(epoch_cache, args.features)
    print(f"Feature matrix: {X_train.shape}\n")

    result = run_experiment(
        X_train, y, groups,
        args.features,
        {'kernel': args.kernel, 'C': args.C}
    )

    print(f"\n{'='*60}")
    print(f"CV Results:")
    print(f"  Features:  {result['features']}")
    print(f"  Kernel:    {result['kernel']}  C: {result['C']}")
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")

    # --- Step 4: Test predictions (optional) ---
    print("\n" + "=" * 60)
    print("STEP 4: Generating Test Set Predictions")
    print("=" * 60)

    print("Training final model on full training set...")
        final_model, final_scaler = train_final_model(
            X_train, y, {'kernel': args.kernel, 'C': args.C})

    print(f"\nLoading test data from '{args.test_path}'...")
        test_data_list, test_subject_ids = load_test_data(args.test_path)

    print("\nExtracting test features...")
        test_cache, test_groups = build_feature_cache_unlabeled(
            test_data_list, test_subject_ids, sfreq)

    print("\nGenerating predictions...")
        predict_test_set(
            final_model, final_scaler,
            test_cache, test_groups,
            args.features, args.output
        )