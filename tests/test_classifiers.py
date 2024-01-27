# Timothy Geiger, acse-tfg22

import pytest
from lung_cancer.models import QNNClassifier, NNClassifier
from lung_cancer.data_handling import DataWrapper

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def dataset():
    """This function creates a dummy dataset.

    Returns:
        pd.DataFrame: Input features.
        pd.Series: Output labels.
    """
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


# parity function
def parity(x):
    return "{:b}".format(x).count("1") % 2


class BasicClassifier(nn.Module):
    """A dummy classifier.
    """
    def __init__(self):
        super(BasicClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class TestQuantumClassifier:
    def test_quantum_classifier_init(self):
        """This function tests the quantum classifier
        initializer.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create classifier
        classifier = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create trainings data
        # trainings data is needed for the quantum
        # model even though no training is carried out
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)
        classifier.train_data_wrapper = data

        # check if attributes are correct
        assert classifier.feature_dimension == 2

        assert isinstance(
            classifier.qnn, SamplerQNN)

        assert isinstance(
            classifier.classifier, NeuralNetworkClassifier)

    def test_quantum_classifier_init_fail(self):
        """This function tests the quantum classifier
        initializer. This function is supposed to
        raise an error.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # check if error gets raised because the parity
        # function was not passed
        with pytest.raises(TypeError):
            QNNClassifier(
                2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2)

    def test_classic_classifier_init(self):
        """This function tests the classic classifier
        initializer.
        """

        # define mode, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # create classifier
        classifier = NNClassifier(model, optimizer, criterion)

        # check if parameters are correct
        assert isinstance(
            classifier.optimizer, optim.Adam)

        assert isinstance(
            classifier.loss, nn.CrossEntropyLoss)

        assert isinstance(
            classifier.classifier, nn.Module)

    def test_classic_classifier_init_fail(self):
        """This function tests the classic classifier
        initializer. This function is supposed to
        raise an error.
        """

        # define model and optimizer
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # check if error gets raised because
        # no loss function was passed
        with pytest.raises(TypeError):
            NNClassifier(model, optimizer)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_train_valid(self):
        """This function tests the train function
        of the quantum classifier. In addiation it
        tests the vlaidation function.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create classifier
        classifier = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the model
        classifier.train(data)

        # check if attribute is correct
        assert isinstance(classifier.train_data_wrapper, DataWrapper)

        # create validaton dataset
        X = pd.DataFrame({
            'feature1': [8, 3, 6, 2],
            'feature2': [4, 4, 2, 2]})
        y = pd.Series([0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # check if output of the validation function is correct
        assert isinstance(classifier.valid(data), float)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_train_valid(self):
        """This function tests the train function
        of the classic classifier. In addiation it
        tests the vlaidation function.
        """

        # define mode, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # create the classifier
        classifier = NNClassifier(model, optimizer, criterion)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the classifier
        classifier.train(2, data)

        # create validaton dataset
        X = pd.DataFrame({
            'feature1': [8, 3, 6, 2],
            'feature2': [4, 4, 2, 2]})
        y = pd.Series([0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # class valid function
        res = classifier.valid(data)

        # check if output of the validation function is correct
        assert isinstance(res, tuple)
        assert isinstance(res[0], torch.Tensor)
        assert isinstance(res[1], float)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_train_valid_fail(self):
        """This function tests the train function
        of the quantum classifier. In addiation it
        tests the vlaidation function. This function
        is supposed to raise an error.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.L1Loss()

        # create the classifier
        classifier = NNClassifier(model, optimizer, criterion)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # check if an error gets raised because
        # the loss function is not yet implemented
        # in my classifier wrapper
        with pytest.raises(ValueError):
            classifier.train(2, data)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_train_callback(self):
        """This function tests the train function
        of the quantum classifier with a callback
        function defined and passed to the training
        process.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # create the classifier
        classifier = NNClassifier(model, optimizer, criterion)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # define a callback function
        def callback(a, b):

            # check if parameters have the correct datatype
            assert isinstance(a, np.ndarray)
            assert isinstance(b, float)

        # train the model and pass the callback function
        classifier.train(2, data, callback)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_predict(self):
        """This function tests the predict function of
        the quantum classifier.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create the classifier
        classifier = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the classifier
        classifier.train(data)

        # create prediction data
        X = np.array([[2, 5],
                      [5, 4],
                      [4, 5]])

        # predict
        preds = classifier.predict(X)

        # checkout if output of the predictin function is correct
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(X)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_predict(self):
        """This function tests the predict function of
        the classic classifier.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # create the classifer
        classifier = NNClassifier(model, optimizer, criterion)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the classifier
        classifier.train(2, data)

        # create prediction data
        X = torch.tensor([
            [2, 5],
            [5, 4],
            [4, 5]], dtype=torch.float32)

        # predict
        preds = classifier.predict(X)

        # check if output has the correct type
        assert isinstance(preds, torch.Tensor)
        assert preds.shape[0] == len(X)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_calc_metrics(self, capsys):
        """This function tests the calc_metrics function of
        the quantum classifier.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create the classfier
        classifier = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the classifier
        classifier.train(data)

        # create predict data
        X = pd.DataFrame({
            'feature1': [0, 1, 2],
            'feature2': [3, 2, 1]})
        y = pd.Series([1, 1, 0])
        data = DataWrapper(X, y, 32, False)

        # predict
        y_pred = classifier.predict(X)

        # calculate metrics
        classifier.calc_metrics(data)
        captured = capsys.readouterr()

        # check if correct auc-score was printed to the screen
        expected = "ROC-AUC:  " + str(roc_auc_score(y, y_pred))
        assert expected in captured.out

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_calc_metrics(self, capsys):
        """This function tests the calc_metrics function of
        the classic classifier.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # create the classifier
        classifier = NNClassifier(model, optimizer, criterion)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the classifier
        classifier.train(2, data)

        # create prediction data
        X = pd.DataFrame({
            'feature1': [0, 1, 2],
            'feature2': [3, 2, 1]})
        y = pd.Series([1, 1, 0])
        data = DataWrapper(X, y, 32, False)

        # calculate the metrics
        classifier.calc_metrics(data)
        captured = capsys.readouterr()

        # create predict data
        X = torch.tensor([
            [0, 3],
            [1, 2],
            [2, 1]], dtype=torch.float32)
        y = torch.tensor([1, 1, 0])

        # predict
        y_pred = classifier.predict(X)

        # check if correct roc-auc score was printed to the screen
        expected = "ROC-AUC:  " + str(roc_auc_score(y, y_pred))
        assert expected in captured.out

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_cross_validate_basic(self, dataset):
        """This function tests the basic cross validation
        functionality of the quantum classifier.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define mode, optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()
        model = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # cross validate
        model.cross_validate(X, y, pipeline, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_cross_validate_with_k(self, dataset):
        """This function tests the cross validation
        functionality of the quantum classifier. In addition
        it uses different values for k.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()
        model = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset
        k_values = [3, 5, 10]

        # iterate over k values and do cross validation
        for k in k_values:
            model.cross_validate(X, y, pipeline, k=k, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_cross_validate_with_weights(self, dataset):
        """This function tests the basic cross validation
        functionality of the quantum classifier. In addition
        it caluclates the class weights for the loss function.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()
        model = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # cross validate with weights
        model.cross_validate(X, y, pipeline, weights=True, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_cross_validate_with_smote(self, dataset):
        """This function tests the basic cross validation
        functionality of the quantum classifier. In addition
        it uses SMOTE to upsample one class.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()
        model = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # cross validate with smote
        model.cross_validate(X, y, pipeline, smote=True, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_cross_validate_with_dim_red(self, dataset):
        """This function tests the basic cross validation
        functionality of the quantum classifier. In addition
        it uses different values for the dimension reduction.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()
        model = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # define different dimension reduction techniques
        dim_red_techniques = ['pca', 'agglo']

        # iterate over reduction techniques and do cross validaton
        for dim_red in dim_red_techniques:
            model.cross_validate(X, y, pipeline, dim_red=dim_red, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_cross_validate_with_kbestNum(self, dataset):
        """This function tests the basic cross validation
        functionality of the quantum classifier. In addition
        it select k best features.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset
        kbest_values = [1, 2, 3]

        # iterate over k best values
        for kbestNum in kbest_values:

            # create classifier
            model = QNNClassifier(
                kbestNum, ZFeatureMap(kbestNum), RealAmplitudes(kbestNum),
                optimizer, loss, 2, parity)

            # cross validate
            model.cross_validate(
                X, y, pipeline, kbest=True,
                kbestNum=kbestNum, dimension=kbestNum)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_cross_validate_basic(self, dataset):
        """This function tests the basic cross validation
        functionality of the classic classifier.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        model = NNClassifier(model, optimizer, criterion)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # cross validate
        model.cross_validate(X, y, pipeline, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_cross_validate_with_k(self, dataset):
        """This function tests the cross validation
        functionality of the classic classifier. In addition
        it uses different values for k.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        model = NNClassifier(model, optimizer, criterion)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset
        k_values = [3, 5, 10]

        # iterate over k values and do cross validation
        for k in k_values:
            model.cross_validate(X, y, pipeline, k=k, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_cross_validate_with_weights(self, dataset):
        """This function tests the basic cross validation
        functionality of the classic classifier. In addition
        it caluclates the class weights for the loss function.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        model = NNClassifier(model, optimizer, criterion)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # cross validate with calculating weights
        model.cross_validate(X, y, pipeline, weights=True, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_cross_validate_with_smote(self, dataset):
        """This function tests the basic cross validation
        functionality of the classic classifier. In addition
        it uses SMOTE to upsample one class.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        model = NNClassifier(model, optimizer, criterion)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # cross validate with smote
        model.cross_validate(X, y, pipeline, smote=True, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_cross_validate_with_dim_red(self, dataset):
        """This function tests the basic cross validation
        functionality of the classic classifier. In addition
        it uses different values for the dimension reduction.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define model, optimizer and loss function
        model = BasicClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        model = NNClassifier(model, optimizer, criterion)

        # create a pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset

        # define different dimension reduction techniques
        dim_red_techniques = ['pca', 'agglo']

        # iterate over dimesnion reduction techniques and cross validate
        for dim_red in dim_red_techniques:
            model.cross_validate(X, y, pipeline, dim_red=dim_red, dimension=2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_classic_classifier_cross_validate_with_kbestNum(self, dataset):
        """This function tests the basic cross validation
        functionality of the classic classifier. In addition
        it select k best features.

        Args:
            dataset (tuple): Tuple which consists of
                features and labels.
        """

        # define loss function and pipeline
        criterion = nn.CrossEntropyLoss()
        pipeline = Pipeline([('scaler', StandardScaler())])

        # get the data
        X, y = dataset
        kbest_values = [1, 2, 3]

        # iterate over values for k
        for kbestNum in kbest_values:

            class DummyClassifier(nn.Module):
                """A dummy classifier for testing the
                classical live plotter.
                """
                def __init__(self):
                    super(DummyClassifier, self).__init__()
                    self.fc1 = nn.Linear(kbestNum, 2)
                    self.fc2 = nn.Linear(2, 2)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)

                    return x

            # define model and optimizer
            model = DummyClassifier()
            optimizer = optim.Adam(
                model.parameters(), lr=0.001, weight_decay=1e-5)
            model = NNClassifier(model, optimizer, criterion)

            # cross validate
            model.cross_validate(
                X, y, pipeline, kbest=True,
                kbestNum=kbestNum, dimension=kbestNum)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_quantum_classifier_predict_proba(self):
        """This function tests the predict_proba function
        of the quantum classifier.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create the classifier
        classifier = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create trainings data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the classifier
        classifier.train(data)

        # create prediction data
        X = np.array([[2, 5],
                      [5, 4],
                      [4, 5]])

        # get the prredictions and probabilities
        probs = classifier.predict_proba(X)

        # check if output datatypes and shapes are correct
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3, 2)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    @pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
    def test_quantum_classifier_feature_importance_agglo(self):
        """This function tests the feature_importance function
        of the quantum classifier using agglo for dimension reduction.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create the classifier
        classifier = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create training data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the model
        classifier.train(data)

        # create the pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # create test data
        X = pd.DataFrame({
            'feature1': [4, 6, 2, 6, 8],
            'feature2': [6, 4, 7, 8, 4],
            'feature3': [1, 3, 4, 6, 3]})
        y = pd.Series([1, 1, 0, 0, 0])

        # get feature importance
        result = classifier.get_feature_importance(
            X, y, pipeline, 2
        )

        # check if output datatype and shape are correct
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    @pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
    def test_quantum_classifier_feature_importance_pca(self):
        """This function tests the feature_importance function
        of the quantum classifier using pca for dimension reduction.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create the classifier
        classifier = QNNClassifier(
            2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

        # create training data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1]})
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the model
        classifier.train(data)

        # create the pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])
        X = pd.DataFrame({
            'feature1': [4, 6, 2, 6, 8],
            'feature2': [6, 4, 7, 8, 4],
            'feature3': [1, 3, 4, 6, 3]})
        y = pd.Series([1, 1, 0, 0, 0])

        # get the feature importance using pca
        result = classifier.get_feature_importance(
            X, y, pipeline, 2, 'pca'
        )

        # check if output datatype and shape are correct
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    @pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
    def test_quantum_classifier_bitstrings(self):
        """This function tests the get_bit_strings_probs function
        of the quantum classifier using agglo for dimension reduction.
        """

        # define optimizer and loss function
        optimizer = COBYLA(1)
        loss = CrossEntropyLoss()

        # create the classifier
        classifier = QNNClassifier(
            3, ZFeatureMap(3), RealAmplitudes(3), optimizer, loss, 2, parity)

        # create the pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])

        # create training data
        X = pd.DataFrame({
            'feature1': [1, 4, 3, 5, 2, 2, 6],
            'feature2': [4, 5, 6, 3, 2, 3, 1],
            'feature3': [4, 5, 4, 4, 3, 9, 3]})
        X = pipeline.fit_transform(X)
        y = pd.Series([1, 1, 0, 0, 0, 1, 1])
        data = DataWrapper(X, y, 32, False)

        # train the model
        classifier.train(data)

        X = pd.DataFrame({
            'feature1': [4, 6, 2, 6, 8],
            'feature2': [6, 4, 7, 8, 4],
            'feature3': [1, 3, 4, 6, 3]})

        X = pipeline.transform(X)

        # get the feature importance using pca
        result = classifier.get_bit_strings_probs(X)

        # check if output datatype and shape are correct
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 8)
