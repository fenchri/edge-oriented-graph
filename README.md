# Connecting the Dots
Source code for the paper "[Connecting the Dots: Document-level Relation Extraction with Edge-oriented Graphs](https://www.aclweb.org/anthology/D19-1498.pdf)" in EMNLP 2019.


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
Results can be reproducable, when using a seed equal to 0.
The model was trained on a GK210GL Tesla K80 GPU.


## Datasets & Pre-processing
Download the datasets.
```
mkdir data
cd data
wget https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip
unzip CDR_Data.zip
mv CDR_DATA CDR
wget https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip
unzip fd4a7409365e.zip
mv alexwuhkucs-gda-extraction-fd4a7409365e GDA
cd ..
```

Pre-trained word embeddings
```
mkdir embeds
wget https://drive.google.com/open?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8
```

First download the GENIA Tagger and Sentence Splitter:
```
$ cd data_processing
$ mkdir common && cd common
$ wget http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz
$ tar xvzf geniass-1.00.tar.gz
$ cd geniass/ && make & cd ..
$ git clone https://github.com/bornabesic/genia-tagger-py.git
$ cd genia-tagger-py && make && cd ..
```

In order to process the datasets, they should first be transformed into the PubTator format.
```
$ sh process_cdr.sh
$ sh process_cdr.sh
```



## Usage
Run the main script from training and testing as follows. Select gpu -1 for cpu.
```
$ cd src/
$ python3 eog.py --config ../configs/parameters_cdr.yaml --train --gpu 0
$ python3 eog.py --config ../configs/parameters_cdr.yaml --test --gpu 0
```

All necessary parameters can be stored in the yaml files inside the configs directory.
The following parameters can be also given as direct inputs as well:
```
$ cd src/
$ python3 eog.py --config ../configs/parameters_cdr.yaml --train --gpu 0 --walks 3 --edges MM ME MS ES SS-ind --context --types --dist
```


### Citation
Please cite the following papers when using this code:

> @inproceedings{christopoulou2019connecting,  
title={Connecting the Dots: Document-level Relation Extraction with Edge-oriented Graphs},  
author={Christopoulou, Fenia and Miwa, Makoto and Ananiadou, Sophia},  
booktitle={},   
publisher={} 
year={2019},
publisher={Association for Computational Linguistics},
pages={},    
}

> @inproceedings{christopoulou2018walk,  
title={A Walk-based Model on Entity Graphs for Relation Extraction},  
author={Christopoulou, Fenia and Miwa, Makoto and Ananiadou, Sophia},  
booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},  
year={2018},  
publisher={Association for Computational Linguistics},  
pages={81--88},  
}

