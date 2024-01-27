# Timothy Geiger, acse-tfg22

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import qutip

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import _bloch_multivector_data

from lung_cancer.data_handling import DataWrapper


'''

This class simplifies the creation of the qubit-plots.
These are used when feature maps are examined.

'''


class QubitPlotter:
    """
    Initializes a QubitPlotter object.

    Args:
        datawrapper (DataWrapper): DataWrapper object containing
            the feature data.
        feature_map (QuantumCircuit): Quantum circuit representing
            the feature map.
        num_qubits (int): Number of qubits in the quantum circuit.
        cols (int, optional): Number of columns in the subplot grid.
            Defaults to 5.
    """
    def __init__(self,
                 datawrapper: DataWrapper,
                 feature_map: QuantumCircuit,
                 num_qubits: int,
                 cols: int = 5):

        self.datawrapper = datawrapper
        self.feature_map = feature_map
        self.num_qubits = num_qubits
        self.cols = cols

    def plot_qubits(self) -> None:
        """
        Plots how a feature map transforms the data onto the bloch
        sphere. This is done by converting the statevector to bloch sphere
        coordinates. The Bloch coordinates are then visualized using the
        Qutip library.

        Returns:
            None
        """

        # feature data
        datapoints = self.datawrapper.get_data()[0]

        # target data
        y = self.datawrapper.get_data()[1]

        # save coords for each row for each qubit
        bloch_coords = [[] for _ in range(self.num_qubits)]

        # iterate over features
        for datapoint in datapoints:

            # create quantum circuit
            qc = QuantumCircuit(self.num_qubits)

            # create feature map
            qc.compose(self.feature_map, inplace=True)

            # set parameters
            qc = qc.assign_parameters(datapoint).decompose()

            # run
            statevector = Statevector(qc)
            data = statevector.data

            # statevector to bloch coords
            coords = _bloch_multivector_data(data)

            # save coords
            for qbit in range(self.num_qubits):
                bloch_coords[qbit].append(coords[qbit])

        # create subplots
        fig, axes = plt.subplots(
            math.ceil(self.num_qubits / self.cols),
            self.cols,
            squeeze=False,
            figsize=(10, math.ceil(self.num_qubits / self.cols * 3)),
            subplot_kw={'projection': '3d'})

        # iterate over qubits and add them to the sublots
        for qubit in range(self.num_qubits):

            # create sphere
            sphere = qutip.Bloch(
                fig=fig,
                axes=axes[math.floor(qubit / self.cols)][qubit % self.cols])

            # add all points for that qubit
            for coord in bloch_coords[qubit]:
                sphere.add_points(coord)

            # set marker color and shape
            sphere.point_color = ['blue' if x == 0 else 'red' for x in y]
            sphere.point_marker = ['o' for _ in y]
            sphere.point_size = [12 for _ in y]
            sphere.point_alpha = [0.25 for _ in y]
            sphere.render()

        # hide unsed plots
        for i in range(self.num_qubits,
                       self.cols*math.ceil(self.num_qubits / self.cols)):
            row = math.floor(i / self.cols)
            col = i % self.cols

            # hide plot but keep legend visible
            axes[row][col].set_axis_off()

        # create custom legend
        legend_elements = []

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker='o',
                color='white',
                markerfacecolor='blue',
                markersize=10,
                label='Class 0'))

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker='o',
                color='white',
                markerfacecolor='red',
                markersize=10,
                label='Class 1'))

        # add legend to the figure
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fancybox=True,
            shadow=True)

        # show qubits
        plt.show()
