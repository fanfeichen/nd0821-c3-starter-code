from numpy import ndarray
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import pytest


class TestModel:
    """A class for model test
    """
    def setup(self) -> None:
        self.cur_path = str(Path(__file__).parent.absolute())

        # Load test data
        with open(self.cur_path + '/../data/test.pkl','rb') as f:
            self.data = pickle.load(f)
        with open(self.cur_path + '/../data/encoder.pkl','rb') as f:
            self.encoder = pickle.load(f)
        with open(self.cur_path + '/../data/lb.pkl','rb') as f:
            self.lb = pickle.load(f)

        self.cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        self.X_test, self.y_test, _, _ = process_data(
            self.data, categorical_features=self.cat_features, label="salary", training=False, encoder=self.encoder, lb=self.lb
            )

        with open(self.cur_path + '/../model/model.pkl','rb') as f:
            self.clf = pickle.load(f)

        self.y_pred = inference(self.clf, self.X_test)

    def test_output_type(self):
        """The model output type should be nparray
        """
        assert isinstance(self.y_pred, np.ndarray)

    def test_output_shape(self):
        """ The y_predict and the true_y should have same data shape """
        assert self.y_test.shape == self.y_pred.shape, "Dropping null changes shape."

    def test_f1(self):
        """f1 should be lager than 0.5
        """
        _, _ ,f1 = compute_model_metrics(self.y_test, self.y_pred)
        assert f1 > 0.5
