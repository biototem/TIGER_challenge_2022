# Lymphocytes_Detection

This subdirectory provides training  scripts for detection in [**TIGER**](https://tiger.grand-challenge.org/Data/)

## Process introduction

When starting the project process, please **make sure that the working directory is the project root directory** .

This project includes three sets of main processes and one set of independent processes, which are described as follows:

## ETL process

The dataloader method scripts which would be used in training and testing in this Detection project  are placed on **./DataReader**. Before training, you shoud split ori data to train/val/test set, we choose a random method(more details on **./DataReader/Data_random_partition.py**), or other mothods you want.

## Data augmentation

An extend training data augmentation method we choose is [HistomicsTK](https://github.com/DigitalSlideArchive/HistomicsTK)'s color_normalization,you can see more details on **./image_processing/Macenko_main.py**

## Training process

The startup code of training process is **./main.py**, edits your real data file path at  **./config** `data_path` then will begin training

## evaluation process

The startup code of evaluation process is **./pred.py**, edits your real data file path at **./config** `data_path` then will begin predicting and evaluating.


