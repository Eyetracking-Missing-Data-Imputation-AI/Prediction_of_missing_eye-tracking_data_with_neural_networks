# Prediction_of_missing_eye-tracking_data_with_neural_networks
This repository contains the source code and data for the paper "Prediction of missing eye-tracking data with neural networks (Haberkamp & Reddix, 2023)"  

## Prerequisites
There is a requirement for both MATLAB and Python3    
* MATLAB was utilized for preprocessing and analysis
* Python3 was utilized to build the deep learning models and generate predictions

## Script Overview
Our project requires the user run each of the 5 files sequentially to impute the missing point of gaze and gaze classification data  
* A_Data_Preparation.m - Preprocesses the raw eye-tracking data to prepare for training the ML models and splits the dataset into training and validation  
* B_TCN_Model.ipynb - Trains a temporal convolutional network to predict the point of gaze from the gaze direction, gaze origin, and eyelid opening values
* C_Classifier.ipynb - Trains a multilayer perceptron to classify the point of gaze coordinates as relevant objects in teh flight smulator world model  
* D_Predictions.ipynb  - Predicts the point of gaze and subsequently classifies it using the two above neural networks
* E_Evaluate_Results.m - Determines the error compared to our ground truth dataset and determineshow much of the corrupted dataset is confidently recovered


