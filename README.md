# Variational-Reasoning-Networks

## Setup

#### get the code

First clone the repo and the submodules:

  `git clone git@github.com:yuyuz/Variational-Reasoning-Networks --recursive`
  
The code depends on the graphnn library, which can be found here: https://github.com/Hanjun-Dai/graphnn

#### build the graphnn library

Under the current project folder, build the graphnn library as instructed in the above link. 

  `cd Variational-Reasoning-Networks/graphnn`
  
Make modifications to the make_common file, such that it can correctly locate your cuda-8 library, and intel MKL and intel TBB libraries. 

  `cp make_common.example make_common` 

Then build the graphnn library:

  `make -j`

#### build main source code

To build the code for text based question answering, do the following:

  `cd Variational-Reasoning-Networks/code_qa_txt`

Make modifications to the Makefile, if necessary

  `cp Makefile.example Makefile`

Then build everything:

  `make -j`

#### link the data folder

First download the data from https://github.com/yuyuz/MetaQA

Then link the root data folder into the top level folder of the project.

  `cd Variational-Reasoning-Networks`
  
  `ln -s path/to/your/data metaQA`

## Play with the code

Below we illustrate with the text based question answering. First navigate to the root code folder:

  `cd Variational-Reasoning-Networks/code_qa_txt`

#### pretraining

We first use 5% of the labeled data to train the posterior inference model, as well as knowledge graph reasoning model. 

  `./init_run.sh`

You can make edits to the script, so as to train with different datasets (vanilla or ntm) and different # hops (1, 2 or 3) for reasoning. 

#### joint training

Then we load the pretrained model dump, and use REINFORCE with variance reduction to jointly train the model. 

  `./run.sh`

Typically ~1000 iterations are enough for the pretraining. 


#### inspect the learned model

The script `vis.sh` is used to visualize the learned inference logic, where `eval.sh` can be used to inspect the topic entity recognition model. In some settings (like ntm with 1-hop reasoning), the jointly learned model can further improve the entity recognition performance after pretraining with 5% data. 



## Reference

If you find the code or data is useful, please cite our work:

@inproceedings{zhang2018variational,
  title={Variational reasoning for question answering with knowledge graph},
  author={Zhang, Yuyu and Dai, Hanjun and Kozareva, Zornitsa and Smola, Alexander J and Song, Le},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}

