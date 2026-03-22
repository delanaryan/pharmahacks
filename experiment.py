#experiemnt.py
import numpy as np
import pandas as pd
from main import (load_data_from_folders, build_feature_cache,
                  build_X, run_experiment)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

FEATURE_COMBINATIONS = [
    ['rbp'],
    ['scc'],
    ['rbp', 'scc'],
    ['rbp', 'scc', 'hjorth'],
    ['rbp', 'scc', 'entropy'],
    ['rbp', 'scc', 'plv'],
    ['rbp', 'scc', 'plv_band'],
    ['rbp', 'scc', 'hjorth', 'entropy'],
    ['rbp', 'scc', 'hjorth', 'plv_band'],
    ['rbp', 'scc', 'hjorth', 'entropy', 'plv_band'],
]

SVM_CONFIGS = [
    {'kernel': 'rbf',    'C': 0.1},
    {'kernel': 'rbf',    'C': 1.0},
    {'kernel': 'rbf',    'C': 10.0},
    {'kernel': 'rbf',    'C': 100.0},
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'linear', 'C': 10.0},
]


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

def run_all_experiments(epoch_cache, y, groups):
    """
    Loops through all feature/SVM combinations.
    Builds X from cache for each combo — no recomputation.
    Returns sorted DataFrame of results.
    """
    total = len(FEATURE_COMBINATIONS) * len(SVM_CONFIGS)
    results = []
    exp_num = 0

    for feat_combo in FEATURE_COMBINATIONS:
        key = '+'.join(feat_combo)

        # Build X once per feature combo, reuse across all SVM configs
        X = build_X(epoch_cache, feat_combo)
        print(f"\nFeature combo: {key} — shape: {X.shape}")

        for svm_params in SVM_CONFIGS:
            exp_num += 1
            print(f"  [{exp_num}/{total}] "
                  f"kernel={svm_params['kernel']} C={svm_params['C']}...",
                  end=' ')

            result = run_experiment(X, y, groups, feat_combo, svm_params)
            results.append(result)

            print(f"Acc: {result['accuracy']:.4f} "
                  f"F1: {result['f1']:.4f} "
                  f"Recall: {result['recall']:.4f}")

    df = pd.DataFrame(results).sort_values('f1', ascending=False)
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    sfreq = 128

    print("=" * 60)
    print("STEP 1: Loading and Denoising")
    print("=" * 60)
    data_list, labels_list, subject_ids_list = load_data_from_folders('training')

    print("\n" + "=" * 60)
    print("STEP 2: Extracting and caching all features (once)")
    print("=" * 60)
    epoch_cache, y, groups = build_feature_cache(
        data_list, labels_list, subject_ids_list, sfreq)

    print("\n" + "=" * 60)
    print(f"STEP 3: Running {len(FEATURE_COMBINATIONS)} x "
          f"{len(SVM_CONFIGS)} = "
          f"{len(FEATURE_COMBINATIONS)*len(SVM_CONFIGS)} experiments")
    print("=" * 60)
    df = run_all_experiments(epoch_cache, y, groups)

    # Save
    df.to_csv('experiment_results.csv', index=False)
    print(f"\nSaved to experiment_results.csv")

    # Top 5
    print("\nTop 5 by F1:")
    print("=" * 60)
    print(df[['features', 'kernel', 'C', 'accuracy',
              'precision', 'recall', 'f1',
              'n_features']].head(5).to_string(index=False))

    # Best
    best = df.iloc[0]
    print(f"\nBest experiment:")
    print(f"  Features:  {best['features']}")
    print(f"  Kernel:    {best['kernel']}  C: {best['C']}")
    print(f"  Accuracy:  {best['accuracy']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  F1:        {best['f1']:.4f}")