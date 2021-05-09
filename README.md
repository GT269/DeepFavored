# DeepFavored
A deep learning based method to identify favored(adaptive) mutations from a region or whole-genome with high performance provided by the dual network structure and the capture of the complex dependencies among component statistics.

See our [manuscript](https://www.) for more details.

### Installation:
```
$conda create -n deepfavored python==3.6.10
$conda activate deepfavored
$pip install -r requiredments.txt
```
### Contents:
This directory contains the following:

    1. DeepFavored.py - interface for training and using DeepFavored
    2. train.py
    3. identify.py
    4. network.py 
    5. utils.py
    6. DeepFavored.json - config file containing the hyper parameters for DeepFavored
    7. requirements.txt - list of the dependencies for running DeepFavored
    8. example/ - example input and output of the training and identifying
            Train_data/
                Eu_EastAs_WestAf/ - example directory containing training data, which just are a little part of the complete dataset(too big to upload) for showing how to run DeepFavored        
            Eu_EastAs_WestAf.df.model/ - directory containing the DeepFavored model that we trained to analyse CEU, CHB and YRI as documented in the [manuscript](https://www.)
            test_data/ - example test data 
            test_data_identified/ - example output of DeepFavored

### Training DeepFavored:
#### Usage:  
\>python DeepFavored.py train

#### Command Line Options:
--config \<string\>: Required, path to the config file suffixed with .json and containing the hyper parameters for DeepFavored

--trainData \<string\>: Required, path to the training data directory

--modelDir \<string\>: Required, path to the directory served to save trained model

##### Content of config file:
```
{    "modelHyparamDict": { #Hyper parameters for defining the architecture of DeepFavored
        "H_hiddenLayer_nodeNum": [  #The number of layers and the number of nodes in each layer for the hidden layer specific to the 'H' classifier
            64,
            64,
            64
        ],
        "H_outLayer_actiFunc": "softmax", #Activation function of the output layer of the 'H' classifier
        "H_outLayer_nodeNum": 2, #The number of nodes in the output layer of the 'H' classifier
        "O_hiddenLayer_nodeNum": [ #The number of layers and the number of nodes in each layer for the hidden layer specific to the 'O' classifier
            16
        ],
        "O_outLayer_actiFunc": "softmax", #Activation function of the output layer of the 'O' classifier
        "O_outLayer_nodeNum": 2, #The number of nodes in the output layer of the 'H' classifier
        "hiddenLayer_actiFunc": "relu", #Activation function for all of the hidden layers
        "sharedLayer_nodeNum": [0] #The number of layers and the number of nodes in each layer for the hidden layer shared by 'H' and 'O' classifier
    },
    "trainDataParamDict": { #Hyper parameters for training data
        "componentStats": [
            "Fst",
            "DDAF",
            "iHS",
            "XPEHH",
            "iSAFE",
            "nSL",
            "DiHH"
        ],
        "favMutNum": 10000,
        "hitchNeutMutNum": 200000,
        "ordNeutMutNum": 200000
    },
    "trainParamDict": { #Hyper parameters for training DeepFavored
        "batch_size": 32, #Number of the mutations in each batch for mini-batch training
        "early_stop_epochs": 15,
        "epochs": 100,
        "hidden_batchnorm": true,
        "hidden_dropout": 0,
        "l1_coef": 0,
        "l2_coef": 0,
        "loss_func": "BinaryCrossentropy",
        "lr": 0.01,
        "optimizer_H": "Adadelta",
        "optimizer_O": "Adam",
        "reduce_lr_epochs": 10,
        "init":"glorot_uniform"
    }
}
```

##### Content of training data directory:
1. Directory hierarchy. The following files and directories must be in a single directory, the path to which will be passed to *DeepFavored.py train* using the flag --trainData.
     >     favored_mutations/  
     >         exampleFile1.tsv
     >         exampleFile2.tsv  
     >         ...  
     >     hitchhiking_neutral_mutations/  
     >         exampleFile1.tsv
     >         exampleFile2.tsv  
     >         ...
     >     ordinary_neutral_mutations/
     >         sweep_sim/
     >             exampleFile1.tsv
     >             exampleFile2.tsv
     >             ...
     >         neutral_sim/
     >             exampleFile1.tsv
     >             exampleFile2.tsv
     >             ...

2. exampleFile.tsv - tab-delimited file with one line per simulated SNP with a favored/hitchhiking neutral/ordinary neutral mutation, up to the directory name. File can have any number of columns for identification, but must have one column for each of the component statistics in the config file. Header line should use the names of the statistics with the same spelling and capitalization as in the config file. The statistics need not be in the same order.

    | POS  | Fst  |XPEHH |  iHS |DDAF  | iSAFE|
    |:----:|:----:|:----:|:----:|:----:|:----:|
    | 1   | 3.4  |  2.2 | -3.4 |  1.3 | 0.1  |
    | ...  | ...  |  ... | ...  |  ... | ...  |

#### Output:
Under the directory specified by *--modelDir*, *DeepFavored.py train* output several files for the trained model

###### Example:
```
$ python DeepFavored.py train --config ./DeepFavored.json
                    --trainData ./example/train_data/Eu_EastAs_WestAf
                    --modelDir ./example/Eu_EastAs_WestAf.df.model
```

### Running DeepFavored on testing data:
#### Usage:  
\>python DeepFavored.py identify

#### Command Line Options:
--model \<string\>: Required, path to the directory served to save trained model

--input <string\>: Required, path to the input, a file like the exampleFile.tsv above or a directory containing this kind of              files.

--outDir \<string\>: Optional, path to the directory saving the output files. If not specified, it will be the same direcory                as the input by default

#### Output:
Each output file(suffixed by *.df.out*) is a TAB separated file in the following format.

| Pos  | DF_H | DF_O |  DF  |
|:----:|:----:|:----:|:----:|
| 102  | 0.9  |  0.8 | 0.72 |
| ...  | ...  |  ... | ...  |

With following headers:
- Pos: Position (bp) sorted in ascending order
- DF_H: Score output by the 'H' classifier
- DF_H: Score output by the 'O' classifier
- DF: DeepFavored score, equal to DF_H times DF_O

#### Examples:  
```
$ python DeepFavored.py identify --modelDir ./example/Eu_EastAs_WestAf.df.model
                      --input ./example/test_data
                      --outDir ./example/test_data_identified
```
