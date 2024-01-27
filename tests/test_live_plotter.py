# Timothy Geiger, acse-tfg22

import pytest
from lung_cancer.data_handling import DataWrapper

from lung_cancer.models import QNNClassifier, NNClassifier
from lung_cancer.visualize import LivePlotter, QuantumLivePlotter, \
    ClassicLivePlotter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss

import matplotlib

# Set the backend to "agg"
matplotlib.use('agg')


class BasicClassifier(nn.Module):
    """A dummy classifier for testing the
    classical live plotter.
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


@pytest.fixture
def live_plotter_classic():
    """This method creates a classical plotter
    and returns it.

    Returns:
        ClassicLivePlotter: The classical plotter object.
    """
    model = BasicClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    classifier = NNClassifier(model, optimizer, criterion)
    return ClassicLivePlotter(classifier)


@pytest.fixture
def live_plotter_classic_val():
    """This method creates a classical plotter
    and returns it. In Addition validation
    data is passed to the live plotter.

    Returns:
        ClassicLivePlotter: The classical plotter object.
    """
    model = BasicClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    classifier = NNClassifier(model, optimizer, criterion)

    # create validation data
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1, 1, 0])
    val = DataWrapper(X, y, 32, False)

    return ClassicLivePlotter(classifier, val)


def parity(x):
    return "{:b}".format(x).count("1") % 2


@pytest.fixture
def live_plotter_quantum():
    """This method creates a quantum plotter
    and returns it.

    Returns:
        QuantumLivePlotter: The quantum plotter object.
    """
    optimizer = COBYLA(1)
    loss = CrossEntropyLoss()

    classifier = QNNClassifier(
        2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

    # create trainings data
    # trainings data is needed for the quantum
    # model even though no training is carried out
    X = pd.DataFrame({'feature1': [1, 4, 3, 5, 2, 2, 6],
                      'feature2': [4, 5, 6, 3, 2, 3, 1]})
    y = pd.Series([1, 1, 0, 0, 0, 1, 1])
    data = DataWrapper(X, y, 32, False)
    classifier.train_data_wrapper = data

    return QuantumLivePlotter(classifier)


@pytest.fixture
def live_plotter_quantum_val():
    """This method creates a quantum plotter
    and returns it. In Addition validation
    data is passed to the live plotter.

    Returns:
        QuantumLivePlotter: The quantum plotter object.
    """
    optimizer = COBYLA(1)
    loss = CrossEntropyLoss()

    classifier = QNNClassifier(
        2, ZFeatureMap(2), RealAmplitudes(2), optimizer, loss, 2, parity)

    # create trainings data
    # trainings data is needed for the quantum
    # model even though no training is carried out
    X = pd.DataFrame({'feature1': [1, 4, 3, 5, 2, 2, 6],
                      'feature2': [4, 5, 6, 3, 2, 3, 1]})
    y = pd.Series([1, 1, 0, 0, 0, 1, 1])
    data = DataWrapper(X, y, 32, False)
    classifier.train_data_wrapper = data

    # create validation data
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.DataFrame({'target': [0, 1, 0]})
    val = DataWrapper(X, y, 32, False)

    return QuantumLivePlotter(classifier, val)


class TestLivePlotter:
    def test_live_plotter_raises_error_with_no_classifier(self):
        """This method tests if the plotters raise an
        exception if no classifier is passed to the
        initializer.
        """
        with pytest.raises(TypeError):
            LivePlotter(None)

        with pytest.raises(TypeError):
            ClassicLivePlotter(None)

        with pytest.raises(TypeError):
            QuantumLivePlotter(None)

    def test_live_plotter_classic_init(self, live_plotter_classic):
        """This method tests if the class initializes the
        values correct.

        Args:
            live_plotter_classic (ClassicLivePlotter): Classic live plottter.
        """
        assert live_plotter_classic.classifier is not None
        assert live_plotter_classic.data_wrapper is None
        assert live_plotter_classic.validate is False
        assert live_plotter_classic.train_loss == []
        assert live_plotter_classic.val_loss == []
        assert live_plotter_classic.train_acc == []
        assert live_plotter_classic.val_acc == []

    def test_live_plotter_classic_val_init(self, live_plotter_classic_val):
        """This method tests if the class initializes the
        values correct. This time validation data is passed
        to the live plotter.

        Args:
            live_plotter_classic_val (ClassicLivePlotter):
                Classic live plottter.
        """

        # these should be now different
        assert live_plotter_classic_val.data_wrapper is not None
        assert live_plotter_classic_val.validate is True

        # the rest should stay the same
        assert live_plotter_classic_val.classifier is not None
        assert live_plotter_classic_val.train_loss == []
        assert live_plotter_classic_val.val_loss == []
        assert live_plotter_classic_val.train_acc == []
        assert live_plotter_classic_val.val_acc == []

    def test_live_plotter_quantum_init(self, live_plotter_quantum):
        """This method tests if the class initializes the
        values correct.

        Args:
            live_plotter_quantum (QuantumLivePlotter): Quantum live plottter.
        """
        assert live_plotter_quantum.classifier is not None
        assert live_plotter_quantum.data_wrapper is None
        assert live_plotter_quantum.validate is False
        assert live_plotter_quantum.train_loss == []
        assert live_plotter_quantum.val_loss == []
        assert live_plotter_quantum.train_acc == []
        assert live_plotter_quantum.val_acc == []

    def test_live_plotter_quantum_val_init(self, live_plotter_quantum_val):
        """This method tests if the class initializes the
        values correct. This time validation data is passed
        to the live plotter.

        Args:
            live_plotter_quantum_val (QuantumLivePlotter):
                Quantum live plottter.
        """

        # these should be now different
        assert live_plotter_quantum_val.data_wrapper is not None
        assert live_plotter_quantum_val.validate is True

        # the rest should stay the same
        assert live_plotter_quantum_val.classifier is not None
        assert live_plotter_quantum_val.train_loss == []
        assert live_plotter_quantum_val.val_loss == []
        assert live_plotter_quantum_val.train_acc == []
        assert live_plotter_quantum_val.val_acc == []

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_live_plotter_classic_update(self, live_plotter_classic):
        """This method tests the update method of the live plotter.

        Args:
            live_plotter_classic (ClassicLivePlotter): Classic live plotter.
        """

        # define loss and accuarcy
        loss = 0.4
        accuarcy = 0.9

        # call update method with the previously defined
        # loss and accuracy
        live_plotter_classic.update(loss, accuarcy)

        # check if the train loss and accuaracy are saved
        assert live_plotter_classic.train_loss == [loss]
        assert live_plotter_classic.train_acc == [accuarcy]

        # call update a second time to see if they
        # stack up and do no replace each other
        live_plotter_classic.update(loss, accuarcy)

        # check if the train loss and accuaracy are saved
        assert live_plotter_classic.train_loss == [loss, loss]
        assert live_plotter_classic.train_acc == [accuarcy, accuarcy]

        plt.close()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_live_plotter_classical_val_update(self, live_plotter_classic_val):
        """This method tests the update method of the live plotter
        with validation data.

        Args:
            live_plotter_classic_val (ClassicLivePlotter):
                Classic live plotter.
        """

        # define loss and accuarcy
        loss = 0.4
        accuarcy = 0.9

        # call update method with the previously defined
        # loss and accuracy
        live_plotter_classic_val.update(loss, accuarcy)

        # check if the train loss and accuaracy are saved
        assert live_plotter_classic_val.train_loss == [loss]
        assert live_plotter_classic_val.train_acc == [accuarcy]

        # check if the val loss and accuaracy are saved
        assert len(live_plotter_classic_val.val_loss) == 1
        assert len(live_plotter_classic_val.val_acc) == 1

        plt.close()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_live_plotter_quantum_update(self, live_plotter_quantum):
        """This method tests the update method of the live plotter
        with validation data.

        Args:
            live_plotter_quantum (QuantumLivePlotter): Quantum live plotter.
        """

        # define loss
        loss = 0.4

        # call update method with dummy weights and the
        # previously defined loss
        live_plotter_quantum.update(
            np.array([5, 2, 5, 4, 3, 4, 5, 3]), loss)

        # check if the train loss and accuaracy are saved
        assert live_plotter_quantum.train_loss == [loss]
        assert len(live_plotter_quantum.train_acc) == 1

        # call update a second time to see if they
        # stack up and do no replace each other
        live_plotter_quantum.update(
            np.array([5, 2, 5, 4, 3, 4, 5, 3]), loss)

        # check if the train loss and accuaracy are saved
        assert live_plotter_quantum.train_loss == [loss, loss]
        assert len(live_plotter_quantum.train_acc) == 2

        plt.close()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_live_plotter_quantum_update_val(self, live_plotter_quantum_val):
        """This method tests the update method of the live plotter
        with validation data.

        Args:
            live_plotter_quantum_val (QuantumLivePlotter):
                Quantum live plotter.
        """

        # define loss
        loss = 0.4

        # call update method with dummy weights and the
        # previously defined loss
        live_plotter_quantum_val.update(
            np.array([5, 2, 5, 4, 3, 4, 5, 3]), loss)

        # check if the train loss and accuaracy are saved
        assert live_plotter_quantum_val.train_loss == [loss]
        assert len(live_plotter_quantum_val.train_acc) == 1

        # check if the val loss and accuaracy are saved
        assert len(live_plotter_quantum_val.val_loss) == 1
        assert len(live_plotter_quantum_val.val_acc) == 1

        plt.close()
