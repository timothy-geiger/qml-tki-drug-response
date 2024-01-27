# Pipelines

This folder contains all the pipelines generated from the preprocessing notebooks. Additionally, the final pipeline is also stored here. However, due to storage constraints, these pipelines are not uploaded to GitHub.

To obtain the final pipeline, follow these steps:

1. Create and activate the environment.

    ```shell
    $ conda env create -f environment.yml
    $ conda activate irp-tfg
    $ pip install -e .
    ```

<br>

2. Ensure that the necessary preprocessing notebooks are executed in the correct order.
3. Once the preprocessing notebooks have been executed, run the final pipeline notebook to obtain the final pipeline.

Please refer to the other readme files on how to execute notebooks or on how to get the data.

<br>

**Note:** For the classical model I use a slightly different pipeline. For the classical pipeline I added a step that reduces the dimension. I noticed that this helped a lot with the classical model but not with the quantum model. This is why I have two different pipelines. If you also want the classical pipeline you need to uncomment all lines in the feature engineering notebook in code-cell 27. After that you need to run the feature engineering notebook -> the scaling notebook -> the pipeline notebook. After that rename the file `final_pipeline.joblib` in the pipeline folder to `final_classical_pipeline.joblib`. 