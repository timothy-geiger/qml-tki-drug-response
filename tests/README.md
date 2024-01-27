# Testing

This folder contains all the tests created for all the functionalities of this python package. The coverage of these tests is at 100%. In order to execute these tests, create and activate the environment.

```shell
$ conda env create -f environment.yml
$ conda activate irp-tfg
$ pip install -e .
```

<br>

Using the following command from the root directory of the project, all pytests can be executed:

```shell
$ pytest test
```

<br>

In order to get the coverage, the following command can be executed from the root directory of the project:

```shell
$ pytest --cov-report term --cov=lung_cancer tests/
```

<br>

In order to get generate a coverage report, the following command can be executed from the root directory of the project:

```shell
$ pytest --cov-report term --cov=lung_cancer tests/ --cov-report html
```
