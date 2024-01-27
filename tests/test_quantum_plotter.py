# Timothy Geiger, acse-tfg22

import pytest
import pandas as pd

from qiskit.circuit.library import ZFeatureMap
from qiskit.circuit import QuantumCircuit

from lung_cancer.data_handling import DataWrapper
from lung_cancer.visualize import QubitPlotter

import matplotlib

# Set the backend to "agg"
matplotlib.use('agg')


@pytest.fixture
def create_qubit_plotter():
    """This method create a QubitPlotter object
    and returns it.

    Returns:
        QubitPlotter: The QubitPlotter object.
    """
    num_qubits = 2

    features = pd.DataFrame([
        [1, 4],
        [6, 6],
        [2, 3],
        [8, 4],
    ])
    labels = pd.DataFrame([1, 0, 0, 0])

    datawrapper = DataWrapper(features, labels, 32, False)
    feature_map = ZFeatureMap(num_qubits)

    return QubitPlotter(datawrapper, feature_map, num_qubits)


class TestQubitPlotter:
    def test_constructor(self, create_qubit_plotter):
        """This method tests wheather the constructor of
        the QubitPlotter is working correctly.

        Args:
            create_qubit_plotter (QubitPlotter): QubitPlotter
                object.
        """
        # Arrange
        datawrapper = create_qubit_plotter.datawrapper
        feature_map = create_qubit_plotter.feature_map
        num_qubits = create_qubit_plotter.num_qubits
        cols = create_qubit_plotter.cols

        # Asserts
        assert isinstance(datawrapper, DataWrapper)
        assert isinstance(feature_map, QuantumCircuit)
        assert isinstance(num_qubits, int)
        assert isinstance(cols, int)

    # ignore userwarning that backend agg is used
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_plot_qubits(self, create_qubit_plotter):
        """This method tests wheather the plotting
        function runs without raising an error.

        Args:
            create_qubit_plotter (QubitPlotter): QubitsPlotter
            object.
        """

        # Create Qubits plotter
        qubit_plotter = create_qubit_plotter

        assert qubit_plotter.plot_qubits() is None
