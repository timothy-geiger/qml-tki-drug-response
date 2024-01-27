# Timothy Geiger, acse-tfg22

import pytest

import numpy as np
import pandas as pd

from lung_cancer.data_handling import DatasetWrapper
from lung_cancer.visualize import CorrMatrixPlotter, \
    DistrubutionPlotter, FacetGridPlotter, StackedBarPlotter

import matplotlib.pyplot as plt

import matplotlib

# Set the backend to "agg"
matplotlib.use('agg')


class TestCorrMatrixPlotter:
    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1], columns=['target'])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']))
    ])
    def test_constructor(self, features, labels):
        """This function tests the initializer of
        the plotter class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = CorrMatrixPlotter(data)

        # Asserts
        assert isinstance(plotter._dataset, DatasetWrapper)

    @pytest.mark.parametrize('features, labels, cols', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]),
         pd.DataFrame([1, 0, 1], columns=['target']),
         [0, 1]),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]),
         pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']),
         [0, 2])
    ])
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_plotting(self, features, labels, cols):
        """This function tests the plotting
        functionality of the class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
            cols (List): Columns to use.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = CorrMatrixPlotter(data)

        plotter.show(cols)

        # Access the underlying matplotlib plot object
        plot_obj = plt.gca()

        plot_title = plot_obj.get_title()
        assert plot_title == "Correlation"

        # Get the data of the plot
        assert np.allclose(
            plot_obj.collections[0].get_array(),
            data[cols].corr().to_numpy().flatten()
        )


class TestDistrubutionPlotter:
    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1], columns=['target'])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']))
    ])
    def test_constructor(self, features, labels):
        """This function tests the initializer of
        the plotter class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = DistrubutionPlotter(data)

        # Asserts
        assert isinstance(plotter._dataset, DatasetWrapper)

    @pytest.mark.parametrize('features, labels, cols', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ], columns=['a', 'b', 'c']),
         pd.DataFrame([1, 0, 1], columns=['target']),
         'a'),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]),
         pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']),
         [0, 2])
    ])
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_plotting(self, features, labels, cols):
        """This function tests the plotting
        functionality of the class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
            cols (Union[List, str]): Columns to use.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = DistrubutionPlotter(data)

        # check if plotting returns without errors
        assert plotter.show(cols) is None


class TestFacetGridPlotter:
    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1], columns=['target'])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']))
    ])
    def test_constructor(self, features, labels):
        """This function tests the initializer of
        the plotter class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = FacetGridPlotter(data)

        # Asserts
        assert isinstance(plotter._dataset, DatasetWrapper)

    @pytest.mark.parametrize('features, labels, cols', [
        (pd.DataFrame([
            ['m', 8, 5],
            ['m', 2, 3],
            ['f', 1, 0],
        ], columns=['a', 'b', 'c']),
         pd.DataFrame([1, 0, 1], columns=['target']),
         ['a', 'b']),
        (pd.DataFrame([
            [1.2, 3.4, 'f'],
            [0.5, 2.8, 'f'],
            [2.3, 1.1, 'm'],
            [0.7, 1.5, 'm'],
            [1.9, 2.6, 'f'],
            [2.0, 1.4, 'd']
        ], columns=['a', 'b', 'c']),
         pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']),
         ['c', 'a'])
    ])
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_plotting(self, features, labels, cols):
        """This function tests the plotting
        functionality of the class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
            cols (List): Columns to use. Has to be
                of length 2. First one is the categorial
                column. The second one the numerical
                column.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = FacetGridPlotter(data)

        # check if plotting returns without errors
        assert plotter.show(cols[0], cols[1], 'right') is None

        # check if plotting returns without errors
        assert plotter.show(cols[0], cols[1], 'left') is None

        # check if plotting returns without errors
        assert plotter.show(cols[0], cols[1], 'middle') is None

        # check if error gets raised if it receives an
        # invalid argument
        with pytest.raises(ValueError):
            plotter.show(cols[0], cols[1], 'top')


class TestStackedBarPlotter:
    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1], columns=['target'])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']))
    ])
    def test_constructor(self, features, labels):
        """This function tests the initializer of
        the plotter class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = StackedBarPlotter(data)

        # Asserts
        assert isinstance(plotter._dataset, DatasetWrapper)

    @pytest.mark.parametrize('features, labels, cols', [
        (pd.DataFrame([
            ['m', 8, 5],
            ['m', 2, 3],
            ['f', 1, 0],
        ], columns=['a', 'b', 'c']),
         pd.DataFrame([1, 0, 1], columns=['target']),
         ['a', 'c']),
        (pd.DataFrame([
            [1.2, 3.4, 'f'],
            [0.5, 2.8, 'f'],
            [2.3, 1.1, 'm'],
            [0.7, 1.5, 'm'],
            [1.9, 2.6, 'f'],
            [2.0, 1.4, 'd']
        ], columns=['a', 'b', 'c']),
         pd.DataFrame([1, 1, 0, 1, 0, 0], columns=['target']),
         ['b', 'a'])
    ])
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_plotting(self, features, labels, cols):
        """This function tests the plotting
        functionality of the class.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output labels.
            cols (List): Columns to use. Has to be
                of length 2. First one is the categorial
                column. The second one the numerical
                column.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        plotter = StackedBarPlotter(data)

        # check if plotting returns without errors
        assert plotter.show(cols[0], cols[1]) is None
