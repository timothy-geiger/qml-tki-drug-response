Timothy Geiger, acse-tfg22

# Lung Cancer Drug Prediction Project

This repository contains the implementation of a quantum model for predicting the reponse of lung cancer patients to drugs (TKIs). The dataset is provided by the Taipei Medical University (TMU) and includes medical details such as medical history, pathological information, genetic makeup, history of therapeutic interventions, and laboratory data like the white blood cell count. In addition to medical-specific details, the dataset also includes personal information about a patient like demographic factors, as well as insights into the patient’s personal history, e.g. smoking and drinking habits. The target variable can be one of two states: positive response to the drug or a negative response. The project focuses on the following key aspects:

- The preprocessing of the dataset.
- The development of a classical model to predict the drug response. A neural network and a random forest classifier were used as the classical methods.
- Development of a quantum model to predict the drug response of lung cancer patients to TKIs. A quantum neural network (QNN) was chosen as the quantum model.
- Comparison between the quantum model and classical models.
- Development of an eXplainable Artificial Intelligence (XAI) method for the quantum model.


## Folder Structure

1. **Python Package** (`lung_cancer`):
   - Contains classes and methods to simplify the visualization of complex plots.
   - Provides a data wrapper to handle data preprocessing, addressing the differing data formats required by qiskit and PyTorch.
   - Includes classifier wrappers with additional methods like live plotting and cross-validation.

<br>

2. **Notebooks** (`notebooks`):
   - Includes the developed python package.
   - Handles the preprocessing of the dataset.
   - Used for the model development of the classical and the quantum model.
   - Shows a new approach that combines XAI and quantum machine learning.

<br>

3. **Pipelines** (`pipelines`):
   - Contains preprocessing pipelines for both the quantum and classical models.
   - Instructions in the `pipelines` folder's README guide how to obtain the pipelines.

<br>

4. **Data** (`data`):
   - Contains the data provided by TMU for the project.
   - Due to privacy concerns, sensitive data is not included in this GitHub repository, in accordance with ethical principles.

<br>

5. **Documentation** (`docs`):
   - This directory contains a comprehensive documentation for this project.
   - A pre-generated PDF version is already included.
   - Instructions in the README within the `docs` directory explains how to generate a HTML documentation or create a new PDF version if needed.

<br>

6. **Tests** (`tests`):
   - Contains unit tests for the Python package, covering 100% of its functionality.

<br>

7. **Workflows** (`.github/workflows`):
   - Contains GitHub Actions workflows.
   - These workflows automate testing, flake8 checks, and compatibility tests on various Python and Linux versions.

<br>

8. **Environment Setup**:
   - The `requirements.txt` file lists all necessary packages and their versions for running the package and notebooks.
   - The `environment.yml` file can be used to create an Anaconda environment using the packages from `requirements.txt`.

<br>

Please refer to the other README files in the subfolders for more informations.

## Getting Started

To use the package, follow these steps:

1. Create a conda environment using.
    ```shell
    $ conda env create -f environment.yml
    ```

<br>

2. Activate the created enviroment.
    ```shell
    $ conda activate irp-tfg
    ```

<br>

3. Install the developed python package using.
    ```shell
    $ pip install -e .
    ```
