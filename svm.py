import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_svm(X_train, y_train, X_test, y_test, groups_test):
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Scale the training features to have mean=0 and variance=1, which is important for SVM performance.

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test) # Scale the test features using the same scaler fitted on the training data to ensure consistency in feature scaling.
    
    # Define SVM model
    model = svm.SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True) # Create an SVM model with RBF kernel, regularization parameter C=1.0, and balanced class weights to handle any class imbalance in the dataset. The probability=True argument allows us to get probability estimates for the predictions, which can be useful for evaluating model performance.
    model.fit(X_train_scaled, y_train.ravel()) # Fit the SVM model to the scaled training data and corresponding labels. The ravel() function is used to convert the label array into a 1D array if it is not already in that format. 

    epoch_predictions = model.predict(X_train_scaled) # Get the predicted labels for the training data using the trained SVM model. This will allow us to evaluate the model's performance on the training set.

    final_results = {}
    unique_subjects = np.unique(groups_test) # Get the unique subject IDs from the groups array to evaluate performance for each subject individually.

    for subject in unique_subjects:
        # Find all epochs belonging to ONE person
        indices = np.where(groups_test == subject[0])
        subject_epochs_preds = epoch_predictions[indices]

        final_prediciton = np.bincount(subject_epochs_preds.astype(int)).argmax()
    return None