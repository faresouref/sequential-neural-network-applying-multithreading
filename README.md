# Machine Learning GUI using Tkinter and Keras

## Overview

This project implements a graphical user interface (GUI) for machine learning tasks, specifically for training and evaluating neural network models using Keras and scikit-learn. The GUI allows users to load their dataset, preprocess it, train models in parallel threads, and evaluate them using various metrics.

## Features

- **Dataset Loading:** Load dataset from a CSV file
- **Data Preprocessing:** Preprocess data using StandardScaler
- **Parallel Processing:** Train neural network models in parallel threads
- **Model Evaluation:** Evaluate model performance using accuracy, precision, recall, F1 score, and confusion matrix
- **Cross-Validation:** Perform cross-validation for additional evaluation

## Requirements

- Python 3.x
- Required Python libraries: `tkinter`, `pandas`, `numpy`, `scikit-learn`, `keras`


## Usage

1. Run the GUI:


2. Click on the "Load Data" button to load your dataset in CSV format.
3. After loading the data, click on the "Start Processing" button to begin training the models.
4. The GUI will display the evaluation metrics including accuracy, precision, recall, F1 score, and confusion matrix.
5. Additionally, cross-validation evaluation metrics will be displayed.

## Implementation Details

### Multithreading

This project utilizes multithreading for parallel processing during the training phase. Each thread is responsible for training a chunk of the dataset independently, speeding up the overall training process. The `MyThread` and `_Thread` classes are used to simulate multithreading.

### Sequential Neural Network Model

The neural network model used in this project is implemented with Keras and follows a sequential architecture. The `create_model()` function defines the model with multiple layers, including input, hidden, and output layers. It is compiled with binary cross-entropy loss and the Adam optimizer.

