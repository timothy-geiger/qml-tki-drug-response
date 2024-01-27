# Timothy Geiger, acse-tfg22

# type hint
from typing import Optional

# abstract methods
from abc import ABC, abstractmethod

# numpy
import numpy as np

# matplotlib
from matplotlib import pyplot as plt

# for clearing the output
from IPython.display import clear_output

# custom imports
from ..models.classifiers import BaseClassifier
from ..data_handling.wrappers import DataWrapper


'''

Qiskit and PyTorch do not provide options by default
to plot the training and valdiation loss and accuracy
curves. Thats why these classes were developed.

'''


class LivePlotter(ABC):
    """An abstract base class for live plotting during training.

    Args:
        classifier (BaseClassifier): The classifier to monitor.
        data_wrapper (Optional[DataWrapper]): The data wrapper for validation
            data. Default is None.
    """
    def __init__(self,
                 classifier: BaseClassifier,
                 data_wrapper: Optional[DataWrapper] = None):

        # check if classifier is not None
        if classifier is None:
            raise TypeError

        self.classifier = classifier

        self.data_wrapper = data_wrapper
        self.validate = (data_wrapper is not None)

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        plt.rcParams["figure.figsize"] = (12, 6)

    @abstractmethod
    def _calculate_metrics(self) -> None:
        raise NotImplementedError('Method is an abstract class.')

    def _update_plot(self) -> None:
        # clear previous plot
        clear_output(wait=True)

        # plot arrays
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Plot')

        # Loss
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.plot(range(len(self.train_loss)), self.train_loss, label='train')

        if self.validate:
            ax1.plot(range(len(self.val_loss)), self.val_loss, label='val')

        # Accuracy
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy")
        ax2.plot(range(len(self.train_acc)), self.train_acc, label='train')

        if self.validate:
            ax2.plot(range(len(self.val_acc)), self.val_acc, label='val')

        plt.legend()
        plt.show()

    def update(self, *args: int, **kwargs: int) -> None:
        self._calculate_metrics(*args, **kwargs)
        self._update_plot()


class QuantumLivePlotter(LivePlotter):
    """This class helps to plot the training and validation loss and
    accuracy during the training process of a quantum classifier.

    Args:
        classifier (BaseClassifier): The classifier to use.
        data_wrapper (Optional[DataWrapper]): The data wrapper for
            validation data. Default is None.
    """
    def __init__(self,
                 classifier: BaseClassifier,
                 data_wrapper: Optional[DataWrapper] = None):

        super().__init__(classifier, data_wrapper)

    def _calculate_metrics(self,
                           weights: np.ndarray,
                           obj_func_eval: float) -> None:

        # training
        _, train_acc = self.classifier.valid(
            self.classifier.train_data_wrapper, weights)
        self.train_loss.append(obj_func_eval)
        self.train_acc.append(train_acc)

        # validation
        if self.validate:
            val_loss, val_acc = self.classifier.valid(
                self.data_wrapper, weights)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)


class ClassicLivePlotter(LivePlotter):
    """This class helps to plot the training and validation loss and
    accuracy during the training process of a classical classifier.

    Args:
        classifier (BaseClassifier): The classifier to use.
        data_wrapper (Optional[DataWrapper]): The data wrapper for
            validation data. Default is None.
    """
    def __init__(self,
                 classifier: BaseClassifier,
                 data_wrapper: Optional[DataWrapper] = None):

        super().__init__(classifier, data_wrapper)

    def _calculate_metrics(self, train_loss, train_acc) -> None:

        # training
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)

        # validation
        if self.validate:
            val_loss, val_acc = self.classifier.valid(self.data_wrapper)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)
