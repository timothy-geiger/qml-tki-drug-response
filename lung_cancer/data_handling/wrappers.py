# Timothy Geiger, acse-tfg22

from __future__ import annotations
from typing import Any, Tuple, List, Union

import pickle
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn import set_config

# pytorch
import torch
from torch.utils.data import TensorDataset, DataLoader

# set ouput from transformers to dataframe
set_config(transform_output='pandas')


class DataWrapper:
    """
    A class that wraps data for training. This wrapper is needed since
    qiskit and pytorch want the data in different formats. Pytorch needs
    the data to be in the pytorch dataloader, where as quskit wants just
    the data (features & targets) itself.

    Args:
        X (pd.DataFrame): Feature values.
        y (pd.Series): Target values.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y = pd.Series([0, 1, 0])
        >>> wrapper = DataWrapper(X, y, batch_size=2, shuffle=True)
    """
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 batch_size: int,
                 shuffle: bool):

        self._X = X
        self._y = y

        # convert X and y to tensors
        tensor_X = torch.tensor(self._X.values, dtype=torch.float32)
        tensor_y = torch.tensor(self._y.values, dtype=torch.int32)

        # create a tensor dataset from the two tensors
        dataset = TensorDataset(tensor_X, tensor_y)

        # finally create a data loader object for pytorch
        self._loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the raw data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the input
                features and target values as numpy arrays.

        Examples:
            >>> wrapper = DataWrapper(X, y, batch_size=2, shuffle=True)
            >>> X_values, y_values = wrapper.get_data()
        """
        return self._X.values, self._y.values

    def get_loader(self) -> DataLoader:
        """
        Get the pytorch data loader object.

        Returns:
            DataLoader: A data loader object for iterating over the data.

        Examples:
            >>> wrapper = DataWrapper(X, y, batch_size=2, shuffle=True)
            >>> loader = wrapper.get_loader()
            >>> for batch_X, batch_y in loader:
            ...     # Use the batched data for training
        """
        return self._loader


class DatasetWrapper:
    """
    A wrapper class for managing datasets. The user has to specify
    either features and labels or the train, test and validation
    datasets to create this object. If the user inputs features and
    labels he can also specfiy how to split the data. The numbers must
    add up to 1.

    Parameters:
        features (pd.DataFrame, optional): Input features. Defaults to None.
        labels (pd.DataFrame, optional): Input labels. Defaults to None.
        train_X (pd.DataFrame, optional): Training features. Defaults to None.
        test_X (pd.DataFrame, optional): Testing features. Defaults to None.
        val_X (pd.DataFrame, optional): Validation features. Defaults to None.
        train_y (pd.DataFrame, optional): Training labels. Defaults to None.
        test_y (pd.DataFrame, optional): Testing labels. Defaults to None.
        val_y (pd.DataFrame, optional): Validation labels. Defaults to None.
        train_size (float, optional): Size of the training data.
            Defaults to 0.65.
        test_size (float, optional): Size of the testing data.
            Defaults to 0.2.
        val_size (float, optional): Size of the validation data.
            Defaults to 0.15.
        batch_size (int, optional): Batch size. Defaults to 64.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y = pd.DataFrame({'target': [0, 1, 0]})
        >>> dataset = DatasetWrapper(X, y)
    """
    def __init__(self,
                 features: pd.DataFrame = None,
                 labels: pd.DataFrame = None,
                 train_X: pd.DataFrame = None,
                 test_X: pd.DataFrame = None,
                 val_X: pd.DataFrame = None,
                 train_y: pd.DataFrame = None,
                 test_y: pd.DataFrame = None,
                 val_y: pd.DataFrame = None,
                 train_size: float = 0.65,
                 test_size: float = 0.2,
                 val_size: float = 0.15,
                 batch_size: int = 64):

        # Input train, test and val data are given
        if features is None and labels is None:
            if train_X is None or test_X is None or val_X is None or \
                    train_y is None or test_y is None or val_y is None:
                raise ValueError(
                    'Either input "features" and "labels" ' +
                    'or train, test and val data')

            else:
                self._train_X = train_X
                self._test_X = test_X
                self._val_X = val_X
                self._train_y = train_y
                self._test_y = test_y
                self._val_y = val_y

        # Input features and labels are given
        elif features is not None and labels is not None:
            if train_X is not None or test_X is not None or \
                val_X is not None or train_y is not None or \
                    test_y is not None or val_y is not None:
                raise ValueError(
                    'Either input "features" and "labels" ' +
                    'or train, test and val data')

            else:
                # split data
                self._train_X, self._test_X, self._train_y, self._test_y = \
                    train_test_split(
                        features,
                        labels,
                        test_size=test_size,
                        random_state=15
                    )

                self._train_X, self._val_X, self._train_y, self._val_y = \
                    train_test_split(
                        self._train_X,
                        self._train_y,
                        train_size=train_size / (train_size + val_size),
                        random_state=15
                    )

        else:
            raise ValueError(
                'Either input "features" and "labels" ' +
                'or train, test and val data')

        if abs(train_size) + abs(test_size) + abs(val_size) != 1:
            raise ValueError(
                'train_size + test_size + valid_size should be equal to 1.')

        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size

        self._generated = False
        self._df = None

        # geneate dataframe
        self._generate_df()

    def __len__(self) -> int:
        if self._generated:
            raise ValueError('Dataset already generated.')

        return len(self._df)

    def __getitem__(self, key: Any) -> Any:
        if self._generated:
            raise ValueError('Dataset already generated.')

        return self._df[key]

    def __getattr__(self, name: str):
        # if '_df' not in vars(self):
        # raise AttributeError

        if self._generated:
            raise ValueError('Dataset already generated.')

        return getattr(self._df, name)

    def _generate_df(self):
        self._df = pd.concat(
            [pd.concat(
                [self._train_X,
                 self._test_X,
                 self._val_X],
                ignore_index=True),
             pd.concat(
                 [self._train_y,
                  self._test_y,
                  self._val_y],
                 ignore_index=True)],
            axis=1)

    def get_column_infos(self, colname: Union[str, int]):
        """
        Prints information about a specified column.

        Args:
            colname (Union[str, int]): Name of the column.

        Returns:
            None

        Examples:
            >>> dataset = DatasetWrapper(X, y)
            >>> dataset.get_column_infos('feature1')
        """

        print('Datatype:', self._df[colname].dtype.name)
        print('Number of null values:', self._df[colname].isnull().sum())
        print('Number of unique values:', len(self._df[colname].unique()))

    def apply_transformer(self,
                          transformer: Any,
                          fit_on: List[str] = ['X'],
                          fit_data: str = 'train',
                          fit_method: str = 'fit',
                          transform_on: List[str] = ['X'],
                          transform_data: List[str] = ['train', 'test', 'val'],
                          transform_method: str = 'transform',
                          cols=None) -> None:
        """
        Applies a specified transformer on the dataset.

        Args:
            transformer: The transformer object.
            fit_on (list, optional): List of data to fit the transformer on.
                Valid is X or/and y. Defaults to ['X'].
            fit_data (str, optional): Data to fit the transformer on.
                Valid is train, test and val. Defaults to 'train'.
            fit_method (str, optional): Method name to fit the transformer.
                Defaults to 'fit'.
            transform_on (list, optional): List of data to transform using the
                transformer. Valid is X or/and y. Defaults to ['X'].
            transform_data (list, optional): List of data subsets to transform.
                Valid is train or/and test or/and val. Defaults to
                ['train', 'test', 'val'].
            transform_method (str, optional): Method name to transform the
                data. Defaults to 'transform'.
            cols (list, optional): List of columns to apply the transformer on.
                Defaults to None (= all columns).

        Returns:
            None

        Raises:
            ValueError: If the dataset is already generated.

        Examples:

            A simple example using a MinMax Scaler. The MinMax Scaler
            should be fitted to the train data und should transform
            the train, test and valdiation data.

        >>> import pandas as pd
        >>> from sklearn.preprocessing import MinMaxScaler
        >>>
        >>> X = pd.DataFrame({
        >>>     'feature1': [1, 2, 3],
        >>>     'feature2': [4, 5, 6]})
        >>> y = pd.DataFrame({'target': [0, 1, 0]})
        >>>
        >>> dataset = DatasetWrapper(
        >>>     features, labels,
        >>>     train_size=0.7,
        >>>     val_size=0.15,
        >>>     test_size=0.15)
        >>> dataset.apply_transformer(MinMaxScaler())

        An Example using SMOTE as an transformer. Only the train data
        should be transformed. However this time not only the features
        but the targets as well. In addition the fit and transform
        function name is different compared to the standart values.

        >>> import pandas as pd
        >>> from imblearn.combine import SMOTE
        >>>
        >>> X = pd.DataFrame({
        >>>     'feature1': [1, 2, 3],
        >>>     'feature2': [4, 5, 6]})
        >>> y = pd.DataFrame({'target': [0, 1, 0]})
        >>>
        >>> dataset = DatasetWrapper(
        >>>     features, labels,
        >>>     train_size=0.7,
        >>>     val_size=0.15,
        >>>     test_size=0.15)
        >>>
        >>> smote = SMOTE(random_state=random_state)
        >>> dataset.apply_transformer(
        >>>     smote,
        >>>     fit_on=[],
        >>>     transform_on=['X', 'y'],
        >>>     transform_data=['train'],
        >>>     transform_method='fit_resample')

        """

        if self._generated:
            raise ValueError('Dataset already generated.')

        # if no columns are specified use all columns
        if cols is None:
            cols = self._train_X.columns

        args_fit = []

        # gets the data that should be used for fitting
        for fit in fit_on:
            args_fit.append(getattr(self, '_' + fit_data + '_' + fit))

        # fits the transformer with the fitting data with the
        # specified fitting method
        if len(args_fit) != 0:
            fitter = getattr(transformer, fit_method)
            fitter(*args_fit)

        # transforms the data that should be transformed
        for trans_data in transform_data:
            args_transform = []

            # get the data
            for trans_on in transform_on:
                args_transform.append(
                    getattr(self, '_' + trans_data + '_' + trans_on))

            # transforms the data
            former = getattr(transformer, transform_method)
            vals = [former(*args_transform)]

            # updates the transformed data
            flattened = []

            for sublist in vals:
                if isinstance(sublist, list) or isinstance(sublist, tuple):
                    flattened.extend(sublist)
                else:
                    flattened.append(sublist)

            for i in range(len(transform_on)):
                setattr(self, '_' + trans_data + '_' + transform_on[i],
                        flattened[i])

        # update/generate the dataframe after applying the transformer
        self._generate_df()

    def generate(self) -> None:
        """
        Generates the dataset.

        Returns:
            None

        Raises:
            ValueError: If the dataset is already generated.

        Examples:
            >>> dataset = DatasetWrapper(
            >>>     features, labels,
            >>>     train_size=0.7,
            >>>     val_size=0.15,
            >>>     test_size=0.15)
            >>> dataset.generate()
        """

        if self._generated:
            raise ValueError('Dataset already generated.')

        self._train_data = DataWrapper(
            self._train_X, self._train_y, self.batch_size, True)

        self._test_data = DataWrapper(
            self._test_X, self._test_y, self.batch_size, False)

        self._val_data = DataWrapper(
            self._val_X, self._val_y, self.batch_size, False)

        self._generated = True

    def get_train(self) -> DataWrapper:
        """
        Returns the training data.

        Returns:
            DataWrapper: The training data.

        Raises:
            ValueError: If the dataset is not yet generated.

        Examples:
            >>> dataset = DatasetWrapper(
            >>>     features, labels,
            >>>     train_size=0.7,
            >>>     val_size=0.15,
            >>>     test_size=0.15)
            >>>
            >>> train = dataset.get_train()
        """

        if not self._generated:
            raise ValueError('Dataset not yet generated.')

        return self._train_data

    def get_test(self) -> DataWrapper:
        """
        Returns the testing data.

        Returns:
            DataWrapper: The testing data.

        Raises:
            ValueError: If the dataset is not yet generated.

        Examples:
            >>> dataset = DatasetWrapper(
            >>>     features, labels,
            >>>     train_size=0.7,
            >>>     val_size=0.15,
            >>>     test_size=0.15)
            >>>
            >>> test = dataset.get_test()
        """

        if not self._generated:
            raise ValueError('Dataset not yet generated.')

        return self._test_data

    def get_val(self) -> DataWrapper:
        """
        Returns the validation data.

        Returns:
            DataWrapper: The validation data.

        Raises:
            ValueError: If the dataset is not yet generated.

        Examples:
            >>> dataset = DatasetWrapper(
            >>>     features, labels,
            >>>     train_size=0.7,
            >>>     val_size=0.15,
            >>>     test_size=0.15)
            >>>
            >>> val = dataset.get_val()
        """

        if not self._generated:
            raise ValueError('Dataset not yet generated.')

        return self._val_data

    def save(self, filename: str) -> None:
        """
        Saves the dataset to a file.

        Args:
            filename (str): Name of the file to save the dataset.

        Returns:
            None

        Examples:
            >>> dataset = DatasetWrapper(
            >>>     features, labels,
            >>>     train_size=0.7,
            >>>     val_size=0.15,
            >>>     test_size=0.15)
            >>>
            >>>  dataset.save('../data/preprocessed/test_data.pkl')
        """

        to_save = {
            'train_X': self._train_X,
            'test_X': self._test_X,
            'val_X': self._val_X,
            'train_y': self._train_y,
            'test_y': self._test_y,
            'val_y': self._val_y,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'batch_size': self.batch_size,
        }

        with open(filename, 'wb') as file:
            pickle.dump(to_save, file)

    @staticmethod
    def load(filename: str) -> None:
        """
        Loads a dataset from a file.

        Args:
            filename (str): Name of the file to load the dataset.

        Returns:
            None

        Examples:
            >>> dataset = DatasetWrapper.load(
            >>>     '../data/preprocessed/test_data.pkl')
        """

        with open(filename, 'rb') as file:
            tmp_dict = pickle.load(file)
            return DatasetWrapper(**tmp_dict)
