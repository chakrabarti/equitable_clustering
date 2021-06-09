This is the codebase for our paper **A New Notion of Individually Fair Clustering: $\alpha$-Equitable *k*-Center**.

Overview
--------
There are three main components to the codebase: the implementation of the clustering algorithms, the main file to execute these implementations, and the main file to generate plots from the execution of the algorithms. Additionally, there is an auxillary component which helps generate/preprocess the data for clustering and time the experiments.

| Component                | Description                                                                                                                                                                                                                                                   | Relevant files                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Algorithm Implementation | Implementation of Algorithms 1, 2, and 3 from our paper as well as relevant modifications for experiments. Also includes implementation of the Hochbaum-Shmoys clustering algorithm as well as the Gonzalez algorithm as implemented by (Brubach et al. 2020) | equitable_clustering.py            |
| Simulation               | Uses the algorithm implementation to generate clusterings and analyze fairness                                                                                                                                                                                | main.py                            |
| Plots                    | Generates plots from the output of the experiments                                                                                                                                                                                                            | generate_plots.py                  |
| Utilities                | Has helper functions for timing code                                                                                                                                                                                                                          | timer.py, data/data_preparation.py |

Dependencies
------------

* numpy==1.20.3
* scipy==1.6.3
* matplotlib==3.4.2
* pandas==1.2.4
* cycler==0.10.0

Data Preparation
----------------

We use (Anderson et al. 2020)'s codebase to prepare our data; all code in `data` is from their codebase. The sample files we used for our experiments can be found in `data/processed` (`data/adult/adult.pkl, data/processed/bank/bank.pkl, data/processed/creditcard/creditcard.pkl, data/processed/census1990/census1990.pkl, data/processed/diabetes/diabetes.pkl`). These each contain at most 25000 points (because that is the maximum number of points we use in our cluster computations), and were generated using `data/data_preparation.py` as provided in their codebase.


Simulation
-------------------

In order to run Alg-AG, Alg-PP, and Pseudo-PoF-Alg for a given value of k, `k_val` and dataset `dataset_name`:

```bash
python --k k_val --sample_file data/processed/dataset_name/dataset_name_1.pkl  

```

In order to allow for quick modification of algorithms without having to run the Hochbaum-Shmoys clustering algorithm, the Hochbaum-Shmoy algorithm's results are stored after running. They have been included here so that the results presented in the paper can be exactly reproduced here.

| Argument         | Functionality                                                                                                                                                                                                                                                                                | Values                                                                                                                                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sample_file      | Name of the sample dataset to be used for the experiment (if generating new samples, note that the sample dataset must start with the dataset type so that the main file can figure out the default name to be used for the output_directory)                                                | `data/processed/adult/adult_1.pkl`<br> `data/processed/bank/bank_1.pkl`<br> `data/processed/creditcard/creditcard_1.pkl`<br> `data/processed/census1990/census1990_1.pkl` <br> `data/processed/diabetes/diabetes_1.pkl` |
| k                | The value of *k* to be used. Defaults to 2.                                                                                                                                                                                                                                                  | Integer                                                                                                                                                                                                                 |
| seed             | The random seed to be used. Defaults to 123 (used in the paper's experiments).                                                                                                                                                                                                               | Integer                                                                                                                                                                                                                 |
| output_directory | Specifies where to dump the assignments and analysis from the simulations. By default will be set to `output` so that experiments for a given dataset `<dataset` will be dumped to `output/<dataset>` where `<dataset>` is one of "adult", "bank", "creditcard", "census1990", or "diabetes" | String                                                                                                                                                                                                                  |


Plots
-------------------------
In order to generate plots for a given list of values of *k*, <list_of_k_vals> and a given list of datasets, <list_of_datasets>:

```bash
python generate_plots.py --k <list_of_k_vals> --dataset <list_of_datasets> 
```

| Argument         | Functionality                                                                                                                                                                       | Values        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| k                | The values of *k* to be plotted. Defaults to [2, 4, 8, 16, 32, 64, 128] (used in the paper's experiments).                                                                          | List[Integer] |
| dataset          | The datasets for which plots should be generated. Defaults to ["adult", "bank", "creditcard", "census1990", "diabetes"] (used in the paper's experiments).                          | List[String]  |
| output_directory | The output directory in which to search for the results of the experiments. Defaults to `output` so that for a given <dataset>, the results are searched for in `output/<dataset>`. | String        |