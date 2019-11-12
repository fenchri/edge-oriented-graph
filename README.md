# Edge-oriented Graph
Source code for the paper "[Connecting the Dots: Document-level Relation Extraction with Edge-oriented Graphs](https://www.aclweb.org/anthology/D19-1498.pdf)" in EMNLP 2019.

<p align="center">
  <img src="./network.svg" height="190">
</p>

### Requirements
- python 3.5 +
- PyTorch 1.1.0
- tqdm
- matplotlib
- recordtype
- orderedyamlload

```
pip3 install -r requirements.txt
```

### Environment
Results can be reproducable, when using a seed equal to 0 and the following settings: GK210GL Tesla K80 GPU, Ubuntu 16.04


## Datasets & Pre-processing
Download the datasets
```
$ mkdir data && cd data
$ wget https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip && unzip CDR_Data.zip && mv CDR_Data CDR
$ wget https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip && unzip fd4a7409365e.zip && mv alexwuhkucs-gda-extraction-fd4a7409365e GDA
$ cd ..
```

Download the GENIA Tagger and Sentence Splitter:
```
$ cd data_processing
$ mkdir common && cd common
$ wget http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz && tar xvzf geniass-1.00.tar.gz
$ cd geniass/ && make && cd ..
$ git clone https://github.com/bornabesic/genia-tagger-py.git
$ cd genia-tagger-py && make
$ cd ../../
```

In order to process the datasets, they should first be transformed into the PubTator format.
```
$ sh process_cdr.sh
$ sh process_gda.sh
```


## Usage
Run the main script from training and testing as follows. Select gpu -1 for cpu mode.  

**CDR dataset**: Train the model on the training set and evaluate on the dev set, in order to identify the best training epoch.
For testing, re-run the model on the union of train and dev (*train+dev_filter.data*) until the best epoch and evaluate on the test set.

**GDA dataset**: Simply train the model on the training set and evaluate on the dev set. Test the saved model on the test set.

In order to ensure the usage of early stopping criterion, use the '*--early_stop*' option.
If during training early stopping is not triggered, the maximum epoch (specified in the config file) will be used.

Otherwise, if you want to train up to a specific epoch, use the '*--epoch epochNumber*' option without early stopping.
The maximum stopping epochs can be defined by the '*--epoch*' option.

For example, in the CDR dataset:
```
$ cd src/
$ python3 eog.py --config ../configs/parameters_cdr.yaml --train --gpu 0 --early_stop       # using early stopping
$ python3 eog.py --config ../configs/parameters_cdr.yaml --train --gpu 0 --epoch 15         # train until the 15th epoch *without* early stopping
$ python3 eog.py --config ../configs/parameters_cdr.yaml --train --gpu 0 --epoch 15 --early_stop  # set both early stop and max epoch

$ python3 eog.py --config ../configs/parameters_cdr.yaml --test --gpu 0
```

All necessary parameters can be stored in the yaml files inside the configs directory.
The following parameters can be also given as direct input as well:
```
usage: eog.py [-h] --config CONFIG [--train] [--test] [--gpu GPU]
              [--walks WALKS] [--window WINDOW] [--edges [EDGES [EDGES ...]]]
              [--types TYPES] [--context CONTEXT] [--dist DIST] [--example]
              [--seed SEED] [--early_stop] [--epoch EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Yaml parameter file
  --train               Training mode - model is saved
  --test                Testing mode - needs a model to load
  --gpu GPU             GPU number
  --walks WALKS         Number of walk iterations
  --window WINDOW       Window for training (empty processes the whole
                        document, 1 processes 1 sentence at a time, etc)
  --edges [EDGES [EDGES ...]]
                        Edge types
  --types TYPES         Include node types (Boolean)
  --context CONTEXT     Include MM context (Boolean)
  --dist DIST           Include distance (Boolean)
  --example             Show example
  --seed SEED           Fixed random seed number
  --early_stop          Use early stopping
  --epoch EPOCH         Maximum training epoch
```


### Citation
Please cite the following papers when using this code:

```
@inproceedings{christopoulou2019connecting,  
title = "Connecting the Dots: Document-level Neural Relation Extraction with Edge-oriented Graphs",  
author = "Christopoulou, Fenia and Miwa, Makoto and Ananiadou, Sophia",  
booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",  
year = "2019",  
publisher = "Association for Computational Linguistics",  
pages = "4927--4938"  
}  

@inproceedings{christopoulou2018walk,  
title = "A Walk-based Model on Entity Graphs for Relation Extraction",  
author = "Christopoulou, Fenia and Miwa, Makoto and Ananiadou, Sophia",  
booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",  
year = "2018",  
publisher = "Association for Computational Linguistics",  
pages = "81--88",  
}
```
