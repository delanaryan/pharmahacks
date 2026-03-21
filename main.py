import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
from sklearn.model_selection import GroupKFold

from proc import extract_features, filter_subjects
from denoise import denoise_eeg

def load_data_from_folders(base_path='training'):
    data_list = []
    labels = []
    subject_ids = []
    
    # Define your classes based on folder names
    # A = 1 (Alzheimer's), C = 0 (Healthy/Control)
    class_mapping = {'AD': 1, 'CN': 0}
    
    for folder_name, label_value in class_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.npy'):
                # Load the raw EEG data [cite: 128]
                file_path = os.path.join(folder_path, file_name)
                data = np.load(file_path)
                data_no_noise = denoise_eeg(data) # Apply the denoising pipeline to the raw EEG data [cite: 128]

                # Use the filename (minus .npy) as the subject_id [cite: 105]
                sub_id = file_name.replace('.npy', '')
                
                data_list.append(data_no_noise)
                labels.append(label_value)
                subject_ids.append(sub_id)
                
    return data_list, labels, subject_ids

def main():
    #==============================#
    #         Configuration        #
    #==============================#

    sfreq = 128

    data_list, labels_list, subject_ids_list = load_data_from_folders('training')

    all_epoch_features = []
    all_epoch_labels = [] 
    all_epoch_groups = [] # Keep track of which subject owns which epoch

    for data, label, sub_id in zip(data_list, labels_list, subject_ids_list):
        # Division into 30-second windows with 15-second overlap [cite: 75]
        window_size = 30 * sfreq 
        step_size = 15 * sfreq 

        for start in range(0, data.shape[1] - window_size + 1, step_size):
            epoch = data[:, start : start + window_size]

            # extract_features returns RBP and SCC as (30, 5, 19) [cite: 64, 70, 77]
            rbp, scc = extract_features(epoch, sfreq) 

            # Average internal time steps to get one vector per 30s window [cite: 71, 72]
            rbp_vec = np.mean(rbp, axis=0).flatten() # Shape: (95,)
            scc_vec = np.mean(scc, axis=0).flatten() # Shape: (95,)

            # Concatenate to create a single feature vector (190 features)
            combined_features = np.hstack([rbp_vec, scc_vec])

            all_epoch_features.append(combined_features)
            all_epoch_labels.append(label)
            all_epoch_groups.append(sub_id)

    X = np.array(all_epoch_features)
    y = np.array(all_epoch_labels)
    groups = np.array(all_epoch_groups)

    #========================#
    #    Cross-validation    #
    #========================#
    # GroupKFold ensures a subject's windows are only in train OR test to avoid leakage [cite: 109]
    gkf = GroupKFold(n_splits=5) 

    subj_actuals = []
    subj_preds = []

    print(f"Starting Training on {len(np.unique(groups))} total subjects...")

    #========================#
    #       Training         #
    #========================#
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        # Standardizing is essential for SVM performance [cite: 109]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train SVM with balanced class weights to handle group distribution [cite: 35, 110]
        model = SVC(kernel='rbf', C=1.0, class_weight='balanced')
        model.fit(X_train, y_train)

        # Predict epoch labels
        preds = model.predict(X_test)

        # Step 3: Subject-Level Evaluation (Majority Voting) 
        for sid in np.unique(groups_test):
            indices = np.where(groups_test == sid)[0]
            subject_votes = preds[indices]

            # Final prediction is the most frequent class across all epochs for that subject [cite: 83]
            final_diagnosis = np.bincount(subject_votes).argmax()

            subj_preds.append(final_diagnosis)
            # Find the actual label for this subject
            actual_label = y_test[indices[0]]
            subj_actuals.append(actual_label)

    # Output metrics recommended by the challenge [cite: 100, 112]
    print("\n" + "="*40)
    print("FINAL RESULTS (Subject-Level)")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(subj_actuals, subj_preds):.4f}")
    print(f"Precision: {precision_score(subj_actuals, subj_preds):.4f}")
    print(f"Recall:    {recall_score(subj_actuals, subj_preds):.4f}")
    print(f"F1 Score:  {f1_score(subj_actuals, subj_preds):.4f}")
    print("\nDetailed Report:")
    print(classification_report(subj_actuals, subj_preds, target_names=['Healthy', 'Alzheimer']))

if __name__ == "__main__":
    main()