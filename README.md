# DL_Assignment3
## Author : Aniket Salunke CS22M013
## Overview
The project is to 
* model sequence-to-sequence learning problems using Recurrent Neural Networks 
* compare different cells such as vanilla RNN, LSTM and GRU 
* understand how attention networks overcome the limitations of vanilla seq2seq models

## Files 
* *predictions_vanilla.txt:* It contains all the words predicted and its actual transliteration on test dataset on the best model without attention.
* *without-attention.ipynb:* Contains code to run sweep with various hyperparameter configurations. This also contain implementation of various cells like RNN, LSTM and GRU. It also contains all the functions which are required to train the model without attention.
* *train.py:* It contains code to train the seq to seq model without attention, as well as code to support the training of model using command line interface. Used *ArgParse* to support this feature.
* *with-attention.ipynb:* Contains code to run sweep with various hyperparameter configurations. This also contain implementation of various cells like RNN, LSTM and GRU. It also contains all the functions which are required to train the model with attention.
 
## Instructions to train and evaluate various models

1. Run the train.py using command line to model without attention and pass parameters which you want to set. Passing parameters is optional, if you don't want to pass parameters then it will take default hyperparameter values to train the model.
Here is one example of the command to run trainWithoutAttention.py and train the model.
`
python train.py -wp "Assignment 3" -we "cs22m013" -es 128 -bs 256 --cell_type "LSTM" --epochs 5
`
2. After running the command, it will print accuracies and loss. It will also log accuracies and loss in wandb.
3. *Pass wandb entity as your wandb username only, otherwise it will give error.*

# Report Link
