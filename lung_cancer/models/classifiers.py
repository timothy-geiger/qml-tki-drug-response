# Timothy Geiger, acse-tfg22

# type hint
from typing import Optional, Callable, Union, Tuple

# abstract methods
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest

# pytorch
import torch

# qiskit
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit.instruction import Instruction
from qiskit.algorithms.optimizers import Optimizer
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions import Loss
from qiskit.utils import algorithm_globals
from qiskit import Aer, execute
from qiskit_aer import AerSimulator

# custom imports
from ..data_handling.wrappers import DatasetWrapper, DataWrapper

random_state = 8

algorithm_globals.random_seed = random_state
np.random.seed(random_state)
torch.manual_seed(random_state)


class BaseClassifier(ABC):
    """Abstract base class for classifiers.

    This class serves as the base class for all classifier implementations.
    Subclasses should override the abstract methods and provide their own
    implementation.

    """

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError('Method is an abstract class.')

    @abstractmethod
    def valid(self) -> Tuple[float, float]:
        raise NotImplementedError('Method is an abstract class.')

    @abstractmethod
    def cross_validate(self) -> None:
        raise NotImplementedError('Method is an abstract class.')

    @abstractmethod
    def predict(self, X: None) -> float:
        raise NotImplementedError('Method is an abstract class.')

    @abstractmethod
    def calc_metrics(self, data_wrapper: DataWrapper) -> None:
        raise NotImplementedError('Method is an abstract class.')


class QNNClassifier(BaseClassifier):
    def __init__(self,
                 feature_dimension: int,
                 feature_map: Union[QuantumCircuit, Instruction],
                 ansatz: Union[QuantumCircuit, Instruction],
                 optimizer: Optimizer,
                 loss: Loss,
                 output_shape: int,
                 parity: Callable,
                 primitive: Optional[Sampler] = None):
        """
        Initializes a Quantum Neural Network (QNN) Classifier.

        Args:
            feature_dimension (int): The dimensionality of the input features.
            output_shape (int): The number of classes for classification.
            feature_map (Union[QuantumCircuit, Instruction]): The feature map
                circuit or instruction for the QNN.
            ansatz (Union[QuantumCircuit, Instruction]): The ansatz circuit or
                instruction for the QNN.
            parity (Callable): The interpret function for the classifier.
            optimizer (Optimizer): The optimizer for training the QNN.
            loss (Loss): The loss function for the QNN.
            qnn_type (str, optional): The type of QNN, either 'sampler' or
                'estimator'. Defaults to 'sampler'.
            primitive (Optional[Union[Sampler, Estimator]], optional): The
                primitive object for the QNN. Defaults to None.

        Raises:
            ValueError: If qnn_type is not 'sampler' or 'estimator'.

        """

        # iterate over parameters and check if they are not None
        args = locals()

        # iterate over parameters
        for arg in list(args.keys()):
            if arg != 'self':

                # set parameter to self.[parameter name]
                setattr(self, arg, args[arg])

        # Define the circuit for the QNN
        self.circuit = QuantumCircuit(self.feature_dimension)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)

        # create sampler object
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            interpret=self.parity,
            output_shape=self.output_shape,
            sampler=primitive
        )

        # create the classfier
        self.classifier = NeuralNetworkClassifier(
            neural_network=self.qnn,
            optimizer=self.optimizer,
            loss=self.loss,
            one_hot=True
        )

    def train(self,
              data_wrapper: DataWrapper,
              callback: Optional[Callable] = None) -> None:
        """
        Train the classifier using the provided data.

        Args:
            data_wrapper (DataWrapper): The data wrapper object containing
                the training data.
            callback (Callable, optional): Optional callback function.
                Defaults to None.

        Returns:
            None
        """

        # set callback function
        self.classifier._callback = callback

        # get the training data
        self.train_data_wrapper = data_wrapper
        X, y = data_wrapper.get_data()

        # train the classifier
        self.classifier.fit(X, y)

    def valid(self,
              data_wrapper: DataWrapper,
              weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Perform validation on the given data using the trained classifier.

        Args:
            data_wrapper (DataWrapper): The data wrapper object containing
                the input features and target labels.
            weights (np.ndarray, optional): Weights for the classifier.
                Defaults to None.

        Returns:
            Tuple[float, float]: A tuple containing the validation loss
                and accuracy.

        """

        X, y = data_wrapper.get_data()

        # If weights are not provided, calculate the score directly
        if weights is None:
            return self.classifier.score(X, y)

        # during training
        # Calculate loss
        X_tmp, y_tmp = self.classifier._validate_input(X, y)
        probs = self.classifier.neural_network.forward(X_tmp, weights)

        valid_loss = float(np.sum(self.loss(probs, y_tmp)) / X_tmp.shape[0])

        # Calculate accuracy
        predictions = np.argmax(probs, axis=1)
        predict = np.zeros(probs.shape)

        for i, v in enumerate(predictions):
            predict[i, v] = 1

        y_pred = self.classifier._validate_output(predict)
        valid_acc = accuracy_score(y, y_pred)

        return valid_loss, valid_acc

    def cross_validate(self,
                       features: pd.DataFrame,
                       labels: pd.Series,
                       pipeline: Pipeline,
                       k: Optional[int] = 5,
                       weights: Optional[bool] = False,
                       smote: Optional[bool] = False,
                       kbest: Optional[bool] = False,
                       kbestNum: Optional[int] = 0,
                       dim_red: Optional[str] = 'pca',
                       dimension: Optional[int] = 6) -> None:
        """
        Perform cross-validation on the given dataset using the
        specified parameters.

        Args:
            features (pd.DataFrame): The input features as a NumPy array.
            labels (pd.Series): The target labels as a NumPy array.
            pipeline (Pipeline): The data preprocessing pipeline.
            k (int, optional): The number of folds for cross-validation.
                Defaults to 5.
            weights (bool, optional): Whether to calculate class weights.
                Defaults to False.
            smote (bool, optional): Whether to apply SMOTE oversampling.
                Defaults to False.
            kbest (bool, optional): Whether to select kbest features.
                Defaults to False.
            kbestNum (int, optional): Number of features to select
                if kbest=True. Defaults to 0.
            dim_red (str, optional): The dimensionality reduction technique.
                Valid is 'pca' and 'agglo'. Defaults to 'pca'.
            dimension (int, optional): The reduced dimension size.
                Defaults to 6.

        Returns:
            None

        """
        accuracies = []
        roc_aucs = []

        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)

        counter = 0

        # Iterate over the folds
        for train_indices, val_indices in kfold.split(features):

            # Split the data into training and validation sets for this fold
            train_features, val_features = \
                features.iloc[train_indices], \
                features.iloc[val_indices]

            train_labels, val_labels = \
                labels.iloc[train_indices], \
                labels.iloc[val_indices]

            # apply preprocessing pipline on current split
            pipeline.fit(train_features, train_labels)
            train_features = pipeline.transform(train_features)
            val_features = pipeline.transform(val_features)

            # calculate weights
            if weights:
                weights_tmp = compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(train_labels),
                    y=train_labels)

                self.loss.weight = torch.tensor(weights_tmp).float()

            # do smote
            if smote:
                smote = SMOTE(random_state=random_state)
                train_features, train_labels = \
                    smote.fit_resample(train_features, train_labels)

            if kbest:
                kbest = SelectKBest(k=kbestNum)
                train_features = \
                    kbest.fit_transform(train_features, train_labels)

                val_features = kbest.transform(val_features)

            # do dimension reduction using pca
            if dim_red == 'pca':
                pca = PCA(dimension, random_state=random_state)
                pca.fit(train_features)
                train_features = pca.transform(train_features)
                val_features = pca.transform(val_features)

            # do dimension reduction using feature agglomeration
            if dim_red == 'agglo':
                agglo = FeatureAgglomeration(dimension)
                agglo.fit(train_features)
                train_features = agglo.transform(train_features)
                val_features = agglo.transform(val_features)

            # train the classifier on the current fold
            wrapper_train = DataWrapper(train_features, train_labels, 64, True)
            self.train(wrapper_train)

            # predict on the validation set
            y_pred = self.predict(
                torch.tensor(val_features.values, dtype=torch.float32))

            # calculate accuracy and ROC-AUC for the current fold
            acc = accuracy_score(val_labels, y_pred)
            roc_auc = roc_auc_score(val_labels, y_pred)

            # store the metrics for this fold
            accuracies.append(acc)
            roc_aucs.append(roc_auc)

            counter += 1

            print('Accuracy (k=' + str(counter) + ')' + str(acc))
            print('ROC-AUC (k=' + str(counter) + ')' + str(roc_auc))
            print()

        # calculate average accuracies and ROC-AUC
        average_accuracies = sum(accuracies) / len(accuracies)
        average_roc_auc = sum(roc_aucs) / len(roc_aucs)

        print('Accuracy: ', average_accuracies)
        print('ROC-AUC: ', average_roc_auc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the targets of features using the
        classifier. It returns the predicted targets.

        Args:
            X (np.ndarray): Features that should be used
                for the classification.

        Returns:
            np.ndarray: Predicted targets.
        """

        return self.classifier.predict(X)

    # used from:
    # https://github.com/qiskit-community/qiskit-machine-learning/blob/c3a783cedb1bc6424370fe9927828c8e07e3413a/qiskit_machine_learning/algorithms/classifiers/neural_network_classifier.py#L141
    def predict_proba(self,
                      X: np.ndarray) -> np.ndarray:
        """Predict the targets of features using the
        classifier. It returns the probability of
        the sampe belonging to class 0.

        Args:
            X (np.ndarray): Features that should be used
                for the classification.

        Returns:
            np.ndarray: Probabilites.
        """

        X, _ = self.classifier._validate_input(X)

        probs = self.classifier._neural_network.forward(
            X,
            self.classifier._fit_result.x)

        predict_ = np.argmax(probs, axis=1)
        predict = np.zeros(probs.shape)

        for i, v in enumerate(predict_):
            predict[i, v] = 1

        return probs

    def get_feature_importance(self,
                               features: pd.DataFrame,
                               labels: pd.Series,
                               pipeline: Pipeline,
                               num_features_red: int,
                               dim_red: str = 'agglo') -> np.ndarray:
        """This function returns the feature importance of
        the trained model.

        Args:
            features (pd.DataFrame): Input features.
            labels (pd.Series): Output labels.
            pipeline (Pipeline): Preprocessing pipeline.
            num_features_red (int): Number of features for
                dimension reduction.
            dim_red (str, optional): Dimension reduction technique.
                Either 'agglo' or 'pca'. Default: 'agglo'.

        Returns:
            np.ndarray: Feature importance of each feature.
                Order is the same as after applying the
                pipeline.
        """

        # Define the range of values to perturb the parameter
        perturbation_values = \
            [0, 0.01, 0.001, -0.001, -0.01]

        feature_importance = []

        # create a dataset
        dataset = DatasetWrapper(
                    features, labels,
                    train_size=0.6, val_size=0.2, test_size=0.2)

        # apply pipeline
        dataset.apply_transformer(pipeline)
        dataset.generate()

        # get number of features after encoding
        num_features = dataset.get_val().get_data()[0].shape[1]

        # iterate over all features
        for i in range(num_features):
            reference_states = []
            importance = 0

            # iterate over all perturbation values
            for perturbation in perturbation_values:

                # create a dataset
                # same train, test, val split as
                # in the training process
                dataset = DatasetWrapper(
                    features, labels,
                    train_size=0.6, val_size=0.2, test_size=0.2)

                # apply pipeline
                dataset.apply_transformer(pipeline)

                # perturbate one feature
                # after encoding and before feature
                # dimension reduction
                dataset._test_X[dataset._test_X.columns[i]] += perturbation
                dataset._test_X[dataset._test_X.columns[i]] %= 1
                dataset._generate_df()

                # reduce feature dimension after
                # changing the feature
                if dim_red == 'agglo':
                    agglo = FeatureAgglomeration(n_clusters=num_features_red)
                    dataset.apply_transformer(agglo)
                    dataset.generate()

                if dim_red == 'pca':
                    pca = PCA(num_features_red, random_state=random_state)
                    dataset.apply_transformer(pca)
                    dataset.generate()

                # iterate over all datapoints
                for idx, data in enumerate(dataset.get_val().get_data()[0]):

                    # combine input parameters for feature map
                    # and trained weights for the ansatz
                    params = np.concatenate((
                        data,
                        self.classifier.weights))

                    # set parameters
                    qc = self.classifier._neural_network._circuit.copy()
                    qc = qc.assign_parameters(params)

                    # get statevector
                    backend = Aer.get_backend('statevector_simulator')
                    job = execute(qc, backend)
                    perturbed_state = job.result().get_statevector()

                    if perturbation == 0:
                        reference_states.append(perturbed_state)

                    else:
                        importance += \
                            abs(reference_states[idx].dot(
                                perturbed_state.conj())) ** 2

            feature_importance.append(
                importance/((len(perturbation_values)-1)*features.shape[0]))

        return feature_importance

    def get_bit_strings_probs(self,
                              features: pd.DataFrame) -> np.ndarray:
        """This function returns the probability of
        each bitstring.

        Args:
            features (pd.DataFrame): Input features.
        Returns:
            np.ndarray: Probabilities of each bitstring.
        """

        # Create an empty list
        new_data = []

        for _, row in features.iterrows():

            # combine input parameters for feature map
            # and trained weights for the ansatz
            params = np.concatenate((
                row.values,
                self.classifier.weights))

            # set parameters
            qc = self.classifier._neural_network._circuit.copy()
            qc = qc.assign_parameters(params)

            # get statevector
            backend = AerSimulator()
            job = execute(qc, backend)
            counts = job.result().get_counts()

            new_data.append(counts)

        df = pd.DataFrame.from_dict(new_data)
        df = df.fillna(0)
        df = (df/df.sum(axis=1).max())

        return df

    def calc_metrics(self, data_wrapper: DataWrapper):
        """Shows different metrics of the classifier using the dataset
        specified as an argument. It shows Accuracy, Recall, Precision,
        ROC-AUC score and a confusion matrix.

        Args:
            data_wrapper (DataWrapper): The dataset that should be used for
                calculating the metrics.

        """

        # get data and predict
        X, y = data_wrapper.get_data()
        y_pred = self.predict(X)

        # calculate metrics
        print(classification_report(y, y_pred))
        print("ROC-AUC: ", roc_auc_score(y, y_pred))
        print(confusion_matrix(y, y_pred))


class NNClassifier(BaseClassifier):
    def __init__(self,
                 classifier,
                 optimizer,
                 loss):
        """
        Initialize the NNClassifier (Neural Network).

        Args:
            classifier (nn.Module): The neural network classifier.
            optimizer (Optimizer): The optimizer for training the classifier.
            loss (Loss): The loss function used for training.

        Returns:
            None
        """

        # iterate over parameters and check if they are not None
        args = locals()

        # iterate over parameters
        for arg in list(args.keys()):
            if arg != 'self':
                # set parameter to self.[parameter name]
                setattr(self, arg, args[arg])

    def _calc_loss(self, output, target, pred):
        if self.loss._get_name() == 'CrossEntropyLoss':
            loss = self.loss(output, target.long())

        else:
            raise ValueError('Loss "' + self.loss._get_name() +
                             '" not supported yet.')

        return loss

    def train(self,
              nepochs: int,
              data_wrapper: DataWrapper,
              callback: Optional[Callable] = None) -> None:
        """
        Trains the classifier for a specified number of epochs using
        the provided data wrapper.

        Args:
            nepochs (int): The number of epochs to train the classifier.
            data_wrapper (DataWrapper): The data wrapper containing
                the training data.
            callback (Optional[Callable], optional): A callback function
                for tracking the training progress. Defaults to None.

        Returns:
            None
        """

        # get the data loader from the data wrapper
        data_loader = data_wrapper.get_loader()

        # iterate over the specified number of epochs
        for _ in range(nepochs):

            # set the classifier in training mode
            self.classifier.train()

            # initialize variables for tracking the training loss and accuracy
            train_loss, train_acc = 0, 0

            # Iterate over the data loader batches
            for inpt, target in data_loader:
                self.optimizer.zero_grad()

                # get predictions of the classifier
                output = self.classifier(inpt)
                pred = output.softmax(dim=1).max(dim=1)[1]

                # clauclate the loss
                loss = self._calc_loss(output, target, pred)
                loss.backward()

                train_loss += loss*inpt.size(0)

                # calculate the accuracy
                train_acc += accuracy_score(
                    target.clone().detach().numpy(),
                    pred.clone().detach().numpy()) * inpt.size(0)

                # Update the optimizer parameters
                self.optimizer.step()

            # Normalize the accumulated loss and accuracy by the dataset size
            train_loss = train_loss / len(data_loader.dataset)
            train_acc = train_acc / len(data_loader.dataset)

            # Invoke the callback function if provided
            if callback is not None:
                callback(train_loss.detach().numpy(), train_acc)

    def valid(self,
              data_wrapper: DataWrapper) -> Tuple[float, float]:
        """
        Perform validation on the given data using the trained classifier.

        Args:
            data_wrapper (DataWrapper): The data wrapper object containing
                the input features and target labels.
            weights (np.ndarray, optional): Weights for the classifier.
                Defaults to None.

        Returns:
            Tuple[float, float]: A tuple containing the validation loss
                and accuracy.

        """

        # get the data loader from the data wrapper
        data_loader = data_wrapper.get_loader()

        # set the classifier in evaluation mode
        self.classifier.eval()

        # initialize variables for tracking the validation loss and accuracy
        valid_loss, valid_acc = 0, 0

        # Disable gradient calculation during validation
        with torch.no_grad():
            for inpt, target in data_loader:

                # forward pass through the classifier
                output = self.classifier(inpt)

                # compute the predicted class labels
                pred = output.softmax(dim=1).max(dim=1)[1]

                # calculate the loss for the validation batch
                loss = self._calc_loss(output, target, pred)

                valid_loss += loss*inpt.size(0)

                valid_acc += accuracy_score(
                    target.clone().detach().numpy(),
                    pred.clone().detach().numpy()) * inpt.size(0)

            # normalize the accumulated loss and accuracy by the dataset size
            valid_loss = valid_loss / len(data_loader.dataset)
            valid_acc = valid_acc / len(data_loader.dataset)

        # return the computed validation loss and accuracy
        return valid_loss, valid_acc

    def cross_validate(self,
                       features: pd.DataFrame,
                       labels: pd.Series,
                       pipeline: Pipeline,
                       iterations: Optional[int] = 100,
                       k: Optional[int] = 5,
                       weights: Optional[bool] = False,
                       smote: Optional[bool] = False,
                       kbest: Optional[bool] = False,
                       kbestNum: Optional[int] = 0,
                       dim_red: Optional[str] = 'pca',
                       dimension: Optional[int] = 6) -> None:
        """
        Perform cross-validation on the given dataset using the
        specified parameters.

        Args:
            features (np.ndarray): The input features as a NumPy array.
            labels (np.ndarray): The target labels as a NumPy array.
            pipeline (Pipeline): The data preprocessing pipeline.
            iterations (int, optional): The number of training iterations.
                Defaults to 100.
            k (int, optional): The number of folds for cross-validation.
                Defaults to 5.
            weights (bool, optional): Whether to calculate class weights.
                Defaults to False.
            smote (bool, optional): Whether to apply SMOTE oversampling.
                Defaults to False.
            kbest (bool, optional): Whether to select kbest features.
                Defaults to False.
            kbestNum (int, optional): Number of features to select
                if kbest=True. Defaults to 0.
            dim_red (str, optional): The dimensionality reduction technique.
                Valid is 'pca' and 'agglo'. Defaults to 'pca'.
            dimension (int, optional): The reduced dimension size.
                Defaults to 6.

        Returns:
            None
        """

        accuracies = []
        roc_aucs = []

        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)

        # Iterate over the folds
        for train_indices, val_indices in kfold.split(features):

            # Split the data into training and validation sets for this fold
            train_features, val_features = \
                features.iloc[train_indices], \
                features.iloc[val_indices]

            train_labels, val_labels = \
                labels.iloc[train_indices], \
                labels.iloc[val_indices]

            # apply preprocessing pipline on current split
            pipeline.fit(train_features, train_labels)
            train_features = pipeline.transform(train_features)
            val_features = pipeline.transform(val_features)

            # calculate weights
            if weights:
                weights_tmp = compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(train_labels),
                    y=train_labels)

                self.loss.weight = torch.tensor(weights_tmp).float()

            # do smote
            if smote:
                smote = SMOTE(random_state=random_state)
                train_features, train_labels = \
                    smote.fit_resample(train_features, train_labels)

            if kbest:
                kbest = SelectKBest(k=kbestNum)
                train_features = \
                    kbest.fit_transform(train_features, train_labels)

                val_features = kbest.transform(val_features)

            # do dimension reduction using pca
            if dim_red == 'pca':
                pca = PCA(dimension, random_state=random_state)
                pca.fit(train_features)
                train_features = pca.transform(train_features)
                val_features = pca.transform(val_features)

            # do dimension reduction using feature agglomeration
            if dim_red == 'agglo':
                agglo = FeatureAgglomeration(dimension)
                agglo.fit(train_features)
                train_features = agglo.transform(train_features)
                val_features = agglo.transform(val_features)

            # train the classifier on the current fold
            wrapper_train = DataWrapper(train_features, train_labels, 64, True)
            self.train(iterations, wrapper_train)

            # predict on the validation set
            y_pred = self.predict(
                torch.tensor(val_features.values, dtype=torch.float32))

            # calculate accuracy and ROC-AUC for the current fold
            acc = accuracy_score(val_labels, y_pred)
            roc_auc = roc_auc_score(val_labels, y_pred)

            # store the metrics for this fold
            accuracies.append(acc)
            roc_aucs.append(roc_auc)

        # calculate average accuracies and ROC-AUC
        average_accuracies = sum(accuracies) / len(accuracies)
        average_roc_auc = sum(roc_aucs) / len(roc_aucs)

        print('Accuracy: ', average_accuracies)
        print('ROC-AUC: ', average_roc_auc)

    def predict(self, X: torch.tensor) -> torch.LongTensor:
        """Predict the targets of features using the
        classifier. It returns the predicted targets.

        Args:
            X (torch.tensor): Features that should be used
                for the classification.

        Returns:
            torch.LongTensor: Predicted targets.
        """
        output = self.classifier(X)
        return output.softmax(dim=1).max(dim=1)[1]

    def calc_metrics(self, data_wrapper: DataWrapper):
        """Shows different metrics of the classifier using the dataset
        specified as an argument. It shows Accuracy, Recall, Precision,
        ROC-AUC score and a confusion matrix.

        Args:
            data_wrapper (DataWrapper): The dataset that should be used for
                calculating the metrics.

        """

        # get the dataloader
        data_loader = data_wrapper.get_loader()

        # sets mode to eval
        self.classifier.eval()

        y = []
        y_pred = []

        # do forward passes and get prediction
        with torch.no_grad():
            for inpt, target in data_loader:
                output = self.classifier(inpt)
                pred = output.softmax(dim=1).max(dim=1)[1]

                y.extend(target.clone().detach().cpu().numpy())
                y_pred.extend(pred.clone().detach().cpu().numpy())

        # calculate metrics
        print(classification_report(y, y_pred))
        print("ROC-AUC: ", roc_auc_score(y, y_pred))
        print(confusion_matrix(y, y_pred))
