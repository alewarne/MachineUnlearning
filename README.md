# Machine Unlearning for Features and Labels

This repository contains code related to the paper [Machine Unlearning for Features and Labels](https://arxiv.org/pdf/2108.11577.pdf) and is structured as follows:


### Code

* The Code for the unleraning strategies is contained in the [Unlearner](Unlearner) folder. The [DNNUnlearner](Unlearner/DNNUnlearner.py) class contains the first- and second order update strategies, all other classes
inherit from it.
* The [Applications](Applications) folder contains some examples how to use the Unlearner classes as discussed in Section 6 of the paper.

### Models

Due to size limitations we publish not every model but some to experiment with.

* The [LSTM](models/LSTM) folder contains a language generation model as described in the paper. The canary sentence has been inserted 29 times and the telephone number that will be predicted is 0123456789.
* The [CNN](models/CNN) folder contains the poisoned CNN model that has been trained on the CIFAR-10 dataset.

### Data

Due to size limitations we did not upload the Drebin and Enron dataset and refer to the original papers. The remaining datasets used for training can be found in the [data](data) folder.

### BibTex

If you found any of this helpful please cite our paper. You may use the following BibTex entry

```
@misc{WarPirWreRie20,
    title={Machine Unlearning of Features and Labels},
    author={Alexander Warnecke and Lukas Pirch and Christian Wressnegger and Konrad Rieck},
    year={2021},
    eprint={2108.11577},
    archivePrefix={arXiv},
    primaryClass={cs.CR}
  }
```