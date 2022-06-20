# Script to train machine learning model.
from ml.data import clean_data, process_data
from ml.model import train_model, inference, compute_model_metrics, compute_slice_metric
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle

# Load and clean raw data
cur_path = str(Path(__file__).parent.absolute())
data = clean_data(cur_path + "/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Save test data to local
with open(cur_path + '/../data/test.pkl','wb') as f:
    pickle.dump(test,f)
with open(cur_path + '/../data/encoder.pkl','wb') as f:
    pickle.dump(encoder,f)
with open(cur_path + '/../data/lb.pkl','wb') as f:
    pickle.dump(lb,f)

# Train and save a model.
clf = train_model(X_train, y_train)
# save
with open(cur_path + '/../model/model.pkl','wb') as f:
    pickle.dump(clf,f)

y_pred = inference(clf, X_test)
_, _, f1 = compute_model_metrics(y_test, y_pred)
print(f"F1 on test data is: {f1}.")

# Output the performance of the model on slices of the data
performance = compute_slice_metric(clf, "workclass", cat_features, encoder, lb, data)
list_of_strings = [ f'{key} : {performance[key]}' for key in performance ]
with open(cur_path + '/slice_output.txt', 'w') as file:
     [ file.write(f'{st}\n') for st in list_of_strings ]