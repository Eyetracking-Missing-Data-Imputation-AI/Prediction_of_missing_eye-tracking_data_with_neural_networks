# Prediction_of_missing_eye-tracking_data_with_neural_networks
This repository contains the source code and data for the paper "Prediction of missing eye-tracking data with neural networks (Haberkamp & Reddix, 2023)"  

## Prerequisites
There is a requirement for both MATLAB and Python3    
* MATLAB was utilized for preprocessing and analysis
* Python3 was utilized to build the deep learning models and generate predictions

## Project Summary  
Eye-tracking datasets report missing data from poor data quality and track loss. 
Typically, a higher data loss occurs when subject movement is not controlled, like in applied research studies. 
Studies with high data loss cannot draw firm conclusions from their results. 
A method for reliably imputing missing data in such datasets would benefit a range of applied research studies. 
In this study, we propose using deep neural networks to impute the lost data (up to 47%) from the existing eye metrics. 
It is unknown if deep neural networks can generate valid predictions by leveraging the data's temporal relationship
and learning a useful representation of the corrupted data. 
Our dataset included eye-tracking data from fourteen aviators performing flight simulation tasks. 
Data were missing for the point of gaze and gaze classification (18.15±12.75%), despite all data existing for the individual eye metrics. 
We trained a temporal convolutional network to predict the point of gaze from each eye's metrics and trained a multilayer perceptron to classify the point of gaze.
We observed a significant reduction in missing data after deep neural network predictions in training (18.38%→1.11%) and validation (16.19%→1.20%) datasets. 
There were also high classification accuracies (ACC) and low mean absolute error (MAE) in training (ACC: 99.93%; MAE:1.05mm) and validation (ACC: 99.99%; MAE:1.20mm) datasets.
Our results indicate that deep neural networks can recover missing eye-tracking data effectively and appropriately model the underlying gaze behavior. 
Deep neural networks may benefit other eye-tracking studies that require denoising, classification, and domain transformations.

## Script Overview
Our project requires the user run each of the 5 files sequentially to impute the missing point of gaze and gaze classification data  
* A_Data_Preparation.m - Preprocesses the raw eye-tracking data to prepare for training the ML models and splits the dataset into training and validation  
* B_TCN_Model.ipynb - Trains a temporal convolutional network to predict the point of gaze from the gaze direction, gaze origin, and eyelid opening values
* C_Classifier.ipynb - Trains a multilayer perceptron to classify the point of gaze coordinates as relevant objects in teh flight smulator world model  
* D_Predictions.ipynb  - Predicts the point of gaze and subsequently classifies it using the two above neural networks
* E_Evaluate_Results.m - Determines the error compared to our ground truth dataset and determineshow much of the corrupted dataset is confidently recovered


