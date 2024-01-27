# Documentation

This folder contains the PDF documentation. In order to regenerate this documentation again follow the following commands:

1. Create and activate the environment.

    ```shell
    $ conda env create -f environment.yml
    $ conda activate irp-tfg
    $ pip install -e .
    ```

<br>

2. Navigate to this folder:
    ```shell
    $ cd docs
    ```

<br>

3. Create a build folder if it does not yet exists:
    ```shell
    $ mkdir build
    ```

<br>

4. Generate the HTML documentation:

    ```shell
    $ sphinx-build -b html source build/html/
    ```

<br>

5. Generate the PDF documentation:

    ```shell
    $ sphinx-build -b latex source build/latex/
    ```

<br>

6. Navigate to this folder:

    ```shell
    $ cd build/latex
    ```

<br>

7. Generate the PDF documentation:

    ```shell
    $ make
    ```