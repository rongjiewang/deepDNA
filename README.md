# deepDNA
a hybrid convolutional and recurrent neural network for compressing compression human mitochondrial genomes

deepDNA, a novel unified model called deepDNA that combines the convolutional neural network (CNN) with the long short-term memory network (LSTM) for compressing human mitochondrial genome sequences. The experiment has shown that out method deepDNA is able to learn sequence local features through a convolutional layer, and to learn higher level representations of long-term sequences dependencies through a long short-term memory network (LSTM) layer. We evaluated the learned genome sequences representations model on human mitochondrial genome sequences compressing tasks and achieved a satisfactory result.

## Install
This is a step by step instruction for installing the deepDNA for python 2.7*.
### Requirements for python modules & versions
* TensorFlow >= 1.9.0
* Keras >= 2.2.0
* biopython >= 1.72

## Data
To verify the validity of our method, 1000 human complete mitochondrial sequences were downloaded from MITOMAP (http://www.mitomap.org) database.

We experimented our method deepDNA on 1000 human complete mitochondrial genome sequences and random split it into three datasets: 70\% training set, 20\% validation set and 10\% test set.


### File function
readSplice.py: data processing file. It randomly select 1000 human complete mitochondrial genome sequences from downloaded MITOMAP dataset and random split it into three parts (70\% training set, 20\% validation set and 10\% test set) and save them into files.

train_dataset.py: train the deepDNA model parameters using training dataset.

test_dataset.py: test the deepDNA model using test dataset.

    
## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Contact
If you have any question, please contact the author rjwang.hit@gmail.com
