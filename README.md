# Alzheimer's Disease Detection Support Vector Machine (EEG)

## Inspiration
We have some experience with other ML models, and wanted to try out an SVM since it seemed well suited for this challenge. 
## What it does it do
Our model classifies patients as either showing signs of Alzheimer's disease or as healthy controls. The classification is done using a support vector machine model (with a redial basis function kernel and a C-parameter (controls trade-off between generalizability and data fitting) equal to 10 ) fased on extracted features from a denoised-version of the patient 19-channel EEG data. 
Measures of classification strength, obtained from 5-fold validation of the training dataset: 
  Accuracy:  0.8113
  Precision: 0.7586
  Recall:    0.8800
  F1:        0.8148
## How we built it
We denoised the data, removed movement artifacts, and extracted spectral features and coherence information from the data. We used Scikit-learn's SVM functionality, alongside Pytorch, Pandas, and Numpy. 
## Challenges we ran into
It was a challenge choosing the model we wanted to use; we considered building a simple neural net, using a CNN model, or some other classifier models. 
## Accomplishments that we're proud of
We are proud of tackling a challenge that was out of our comfort zone. We worked with naturalistic data, and challenges related to the noisiness of the EEG were rewarding to work through. 
## What we learned
We can accurately separate AD and CN data from a realtively limitted set of spectral features! Spicebros is so good!
## What's next for Alzheimer's Disease Detection Support Vector Machine (EEG)
We would like to experiment with different features and feature extraction methods to improve the compute of our program. 
