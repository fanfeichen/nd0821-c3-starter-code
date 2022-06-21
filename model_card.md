# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Fanfei Chen created the model. It is random forest using the default hyperparameters in scikit-learn 0.24.2.

## Intended Use
This model should be used to predict the salary of a person based off a categorical variables.

## Training Data
The data was obtained from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 

The original data set has 30162 rows, and a 80-20 split was used to break this into a train and test set. We dropped all missing data. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data
The model is evaludated on slices of the data. We iterate through the distinct values in a given categorical variable "workclass".

## Metrics
F1 score is used for evaluating the model performance. The F1 score on testing data is 0.65.
The metrics on slices of the data is in file slice_output.txt. They are as follows:
State-gov : 0.7592592592592593
Self-emp-not-inc : 0.6570281124497992
Private : 0.6519845713574717
Federal-gov : 0.6531986531986532
Local-gov : 0.6848319709355133
Self-emp-inc : 0.786096256684492
Without-pay : 1.0

## Ethical Considerations
- Any type of communication in relation to the research should be done with honesty and transparency.
- Any type of misleading information, as well as representation of primary data findings in a biased way must be avoided.

## Caveats and Recommendations
Working in a command line environment is recommended for ease of use with Git and DVC. If on Windows, WSL1 or 2 is recommended. The requirements.txt file is for creating a conda project environment.
