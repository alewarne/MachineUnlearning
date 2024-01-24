# Machine Unlearning of Features and Labels

This repository contains code related to the paper [Machine Unlearning of Features and Labels](https://arxiv.org/pdf/2108.11577.pdf) published at NDSS 2023 and is structured as follows:

### Setup

* We tested the code with `python3.7.7`
* We recommend setting up a virtual environment (e.g. using [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html))
* Install depdendencies via `pip install -r requirements.txt`
* Install optional dependencies (notebooks etc.) via `pip install -r opt_requirements.txt`
* For the backdoor experiments, have a look at `example_notebooks/Cifar_data.iypnb`. This notebook shows how to setup the data as expected by our backdoor experiments.

### Code

* The Code for the unleraning strategies is contained in the [Unlearner](Unlearner) folder. The [DNNUnlearner](Unlearner/DNNUnlearner.py) class contains the first- and second order update strategies, all other classes
inherit from it.
* The [Applications](Applications) folder contains some examples how to use the Unlearner classes as discussed in Section 6 of the paper.

### Models

Due to size limitations we publish not every model but some to experiment with.

* The [LSTM](models/LSTM) folder contains two language generation model as described in the paper. The canary sentence has been inserted 8 and 29 times respectively and the telephone number that will be predicted is 0123456789.
* The [CNN](models/CNN) folder contains the poisoned CNN model that has been trained on the CIFAR-10 dataset.

### Example Usage

We provide [examples](example_notebooks) to reproduces the results from the paper in jupyter notebooks.

### Data

Due to size limitations we did not upload the raw data for the Drebin and Enron dataset and refer to the original papers instead. The vector representations to run the experiments are given instead. All of them can be found in the [train_test_data](train_test_data) folder.

### BibTex

If you found any of this helpful please cite our paper. You may use the following BibTex entry

```
@inproceedings{WarPirWreRie20,
    title={Machine Unlearning of Features and Labels},
    author={Alexander Warnecke and Lukas Pirch and Christian Wressnegger and Konrad Rieck},
    year={2023},
    booktitle={Proc. of the 30th Network and Distributed System Security (NDSS)}
  }
```
