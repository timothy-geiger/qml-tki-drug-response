# Timothy Geiger, acse-tfg22

import pytest
from lung_cancer.data_handling import DatasetWrapper, DataWrapper

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

import torch
from torch.utils.data import DataLoader


@pytest.fixture
def temp_file(tmpdir):
    filename = os.path.join(tmpdir, 'test_data.pkl')
    yield filename
    os.remove(filename)


class TestDataWrapper:
    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [1]
        ]), pd.DataFrame([1])),
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_creation_valid(self, features, labels):
        """This function tests for valid input parameters.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DataWrapper(features, labels, 32, True)

        # check if the values are the same
        assert np.allclose(data._X, features)
        assert np.allclose(data._y, labels)

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            ['Peter', 3.4, 2.1],
            ['Hannah', 2.8, 1.9],
            ['Jones', 1.1, 0.9],
            ['Lea', 1.5, 0.3],
            ['Eva', 2.6, 2.2],
            ['Tim', 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0])),
        (pd.DataFrame([
            [2.4, 3.4, 2.1],
            [45.1, 2.8, 1.9],
            [1.4, 1.1, 0.9],
        ]), pd.DataFrame(["Jones", 1, 4]))
    ])
    def test_data_wrapper_creation_not_valid(self, features, labels):
        """This function tests for not valid input
        parameters.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # check if exception gets raised
        with pytest.raises(TypeError):
            DataWrapper(features, labels, 32, True)

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [1]
        ]), pd.DataFrame([1])),
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_data_wrapper_get_data_type(self, features, labels):
        """This method tests if the get_data()
        method return the correct type.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DataWrapper(features, labels, 32, True)

        # get the data back
        X, y = data.get_data()

        # check if data is from correcttype
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [1]
        ]), pd.DataFrame([1])),
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_data_wrapper_get_data(self, features, labels):
        """This method tests if the get_data()
        methods returns the values as expected.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DataWrapper(features, labels, 32, True)

        # get the data back
        X, y = data.get_data()

        # check if data is still the same
        assert np.allclose(X, features.values)
        assert np.allclose(y, labels.values)

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [1]
        ]), pd.DataFrame([1])),
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_data_wrapper_get_loader_type(self, features, labels):
        """This method tests if the get_loader()
        methods returns the correct type.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DataWrapper(features, labels, 32, False)

        # get the data loader object
        loader = data.get_loader()

        # check if loader is from correct instance
        assert isinstance(loader, DataLoader)

    @pytest.mark.parametrize(
        'features, labels, batch_size, expected_batches', [
            (
                pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
                pd.DataFrame([0, 1, 0]),
                2,
                [
                    (
                        torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
                        torch.tensor([[0], [1]], dtype=torch.int32),
                    ),
                    (
                        torch.tensor([[5, 6]], dtype=torch.float32),
                        torch.tensor([[0]], dtype=torch.int32),
                    )
                ]
            ),
            (
                pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
                pd.DataFrame([0, 1, 0]),
                3,
                [
                    (
                        torch.tensor([[1, 2], [3, 4], [5, 6]],
                                     dtype=torch.float32),
                        torch.tensor([[0], [1], [0]], dtype=torch.int32),
                    )
                ]
            ),
            (
                pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
                pd.DataFrame([0, 1, 0]),
                1,
                [
                    (
                        torch.tensor([[1, 2]], dtype=torch.float32),
                        torch.tensor([[0]], dtype=torch.int32),
                    ),
                    (
                        torch.tensor([[3, 4]], dtype=torch.float32),
                        torch.tensor([[1]], dtype=torch.int32),
                    ),
                    (
                        torch.tensor([[5, 6]], dtype=torch.float32),
                        torch.tensor([[0]], dtype=torch.int32),
                    )
                ]
            ),
        ])
    def test_data_wrapper_get_loader_batches(
            self, features, labels, batch_size, expected_batches):
        """This method tests if the get_loader() method produces
        the correct batches.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data wrapper object
        wrapper = DataWrapper(
            features, labels, batch_size=batch_size, shuffle=False)

        # get loader
        loader = wrapper.get_loader()

        # iterate over batches
        for i, (batch_X, batch_y) in enumerate(loader):
            expected_batch_X, expected_batch_y = expected_batches[i]

            # check if features are the same
            assert torch.all(torch.eq(batch_X, expected_batch_X))

            # Check batch targets
            assert torch.all(torch.eq(batch_y, expected_batch_y))


class TestDatasetWrapper:
    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_dataset_wrapper_creation_valid(self, features, labels):
        """This function tests for valid input parameters
        when the input only consits of features and labels

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DatasetWrapper(features, labels)

        expected = pd.concat([features, labels], ignore_index=True, axis=1)

        # check if the values are the same
        assert np.allclose(data._df.shape, expected.shape)

        # check if sub sets were generated
        assert data._train_X is not None
        assert data._test_X is not None
        assert data._val_X is not None
        assert data._train_y is not None
        assert data._test_y is not None
        assert data._val_y is not None

        # check length
        assert len(data) == len(features)

        # accessing elements
        assert len(data[0]) == len(features)

        # check accessing attribute of df
        assert len(data.columns) == len(features.columns) + 1

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_dataset_wrapper_valid_split(self, features, labels):
        """This function tests for valid input parameters
        when the input consits of train, test and val split.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create split
        train_size = 0.65
        test_size = 0.2
        val_size = 0.15

        train_X, test_X, train_y, test_y = \
            train_test_split(
                features,
                labels,
                test_size=test_size,
                random_state=15
            )

        train_X, val_X, train_y, val_y = \
            train_test_split(
                train_X,
                train_y,
                train_size=train_size / (train_size + val_size),
                random_state=15
            )

        # create data object
        data = DatasetWrapper(
            train_X=train_X, test_X=test_X, val_X=val_X,
            train_y=train_y, test_y=test_y, val_y=val_y,
            train_size=train_size, test_size=test_size, val_size=val_size)

        expected = pd.concat([features, labels], axis=1)

        # check if the values are the same
        assert np.allclose(data._df.shape, expected.shape)

        # check if sub sets were generated
        assert data._train_X is not None
        assert data._test_X is not None
        assert data._val_X is not None
        assert data._train_y is not None
        assert data._test_y is not None
        assert data._val_y is not None

        # check length
        assert len(data) == len(features)

        # accessing elements
        assert len(data[0]) == len(features)

        # check accessing attribute of df
        assert len(data.columns) == len(features.columns) + 1

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_dataset_wrapper_not_valid_all(self, features, labels):
        """This function tests for invalid input parameters
        when to much arguments are passed t the initializer.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create split
        train_size = 0.65
        test_size = 0.2
        val_size = 0.15

        train_X, test_X, train_y, test_y = \
            train_test_split(
                features,
                labels,
                test_size=test_size,
                random_state=15
            )

        train_X, val_X, train_y, val_y = \
            train_test_split(
                train_X,
                train_y,
                train_size=train_size / (train_size + val_size),
                random_state=15
            )

        # check if exception gets raised
        with pytest.raises(ValueError):
            DatasetWrapper(
                features, labels,
                train_X, test_X, val_X,
                train_y, test_y, val_y)

        with pytest.raises(ValueError):
            DatasetWrapper(
                features=features,
                train_X=train_X)

    def test_dataset_wrapper_not_valid_empty(self):
        """This function tests for invalid input parameters
        when no arguments are passed to the initializer.
        """

        # check if exception gets raised
        with pytest.raises(ValueError):
            DatasetWrapper()

    @pytest.mark.parametrize('train_size, test_size, val_size', [
        (0.3, 0.3, 0.3),
        (0.4, 0.3, 0.4),
        (0, 0, 0),
        (1, -1, 0)
    ])
    def test_dataset_wrapper_not_valid_sizes(
            self, train_size, test_size, val_size):
        """This function tests for invalid input parameters
        when no arguments are passed to the initializer.

        Args:
            train_size (Union[int, double]): Size if the train split.
            test_size (Union[int, double]: Size if the test split.
            val_size (Union[int, double]: Size if the val split.
        """

        features = pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ])

        labels = pd.DataFrame([1, 0, 1])

        # check if exception gets raised
        with pytest.raises(ValueError):
            DatasetWrapper(features, labels,
                           train_size=train_size,
                           test_size=test_size,
                           val_size=val_size)

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_dataset_wrapper_generate(self, features, labels):
        """This function tests the generate function.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        data.generate()

        # check length
        with pytest.raises(ValueError):
            len(data) == len(features)

        # accessing elements
        with pytest.raises(ValueError):
            len(data[0]) == len(features)

        # check accessing attribute of df
        with pytest.raises(ValueError):
            len(data.columns) == len(features.columns) + 1

        # accessing elements
        with pytest.raises(ValueError):
            data.generate()

        # create data object
        # apply pca
        with pytest.raises(ValueError):
            pca = PCA(2)
            data.apply_transformer(pca)

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
    def test_dataset_wrapper_summary(self, capsys, features, labels):
        """This function tests the summary function.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        data.get_column_infos(0)

        captured = capsys.readouterr()

        expected = 'Datatype: ' + features[0].dtype.name + '\n' + \
            'Number of null values: 0\n' + \
            'Number of unique values: ' + str(len(features[0].unique())) + '\n'

        assert captured.out == expected

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_dataset_wrapper_getter(self, features, labels):
        """This function tests the getter functions.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        data.generate()

        assert data.get_train() is not None
        assert data.get_test() is not None
        assert data.get_val() is not None

        assert isinstance(data.get_train(), DataWrapper)
        assert isinstance(data.get_test(), DataWrapper)
        assert isinstance(data.get_val(), DataWrapper)

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_dataset_wrapper_getter_invalid(self, features, labels):
        """This function tests the getter functions.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        data = DatasetWrapper(features, labels)

        # check if exception gets raised
        with pytest.raises(ValueError):
            data.get_train()

        # check if exception gets raised
        with pytest.raises(ValueError):
            data.get_test()

        # check if exception gets raised
        with pytest.raises(ValueError):
            data.get_val()

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [2, 1, 0],
        ]), pd.DataFrame([1, 0, 1])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0]))
    ])
    def test_dataset_wrapper_save_load(self, features, labels, temp_file):
        """This function tests the getter functions.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
            temp_file (File): A temporary file.
        """

        # create data object
        data = DatasetWrapper(features, labels)
        data.save(temp_file)

        # Load the dataset from the temporary file
        loaded_dataset = DatasetWrapper.load(temp_file)

        # Assert that the loaded dataset has the
        # same attributes as the original dataset
        assert np.allclose(loaded_dataset._train_X, data._train_X)
        assert np.allclose(loaded_dataset._test_X, data._test_X)
        assert np.allclose(loaded_dataset._val_X, data._val_X)
        assert np.allclose(loaded_dataset._train_y, data._train_y)
        assert np.allclose(loaded_dataset._test_y, data._test_y)
        assert np.allclose(loaded_dataset._val_y, data._val_y)
        assert loaded_dataset.train_size == data.train_size
        assert loaded_dataset.test_size == data.test_size
        assert loaded_dataset.val_size == data.val_size
        assert loaded_dataset.batch_size == data.batch_size

    @pytest.mark.parametrize('features, labels', [
        (pd.DataFrame([
            [5, 8, 5],
            [2, 2, 3],
            [4, 1, 0],
            [2, 1, 2],
            [4, 7, 4],
            [12, 1, 5],
            [2, 10, 2],
            [1, 3, 0],
            [2, 10, 2],
            [1, 5, 0],
            [5, 2, 0],
            [5, 3, 2]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                         columns=['target'])),
        (pd.DataFrame([
            [1.2, 3.4, 2.1],
            [0.5, 2.8, 1.9],
            [2.3, 1.1, 0.9],
            [0.7, 1.5, 0.3],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7],
            [1.9, 2.6, 2.2],
            [2.0, 1.4, 1.7]
        ]), pd.DataFrame([1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                         columns=['target']))
    ])
    def test_dataset_apply_transformer(self, features, labels):
        """This function tests the apply transformer function.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.DataFrame): Output targets.
        """

        # create data object
        # apply pca
        data = DatasetWrapper(features, labels)
        pca = PCA(2)
        data.apply_transformer(pca)

        assert len(data.columns) == 2 + 1

        # create data object
        # apply pca on specific columns
        data = DatasetWrapper(features, labels)
        pca = PCA(1)
        data.apply_transformer(pca, cols=['0', '1'])

        assert len(data.columns) == features.shape[1]-1

        # create data object
        # apply smote
        data = DatasetWrapper(features, labels)
        smote = SMOTE(k_neighbors=2)
        data.apply_transformer(
            smote,
            fit_on=[],
            transform_on=['X', 'y'],
            transform_data=['train'],
            transform_method='fit_resample')

        assert len(data) != len(labels)
