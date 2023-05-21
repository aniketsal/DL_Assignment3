# Importing Libraries

from io import open
import unicodedata
import string
import re
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import argparse
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd

# Load Training, Validation, and Test Data

df = pd.read_csv('aksharantar_sampled/hin/hin_train.csv',names = ["English",'Hindi'],header = None)
df_val=pd.read_csv('aksharantar_sampled/hin/hin_valid.csv',names=["English","Hindi"],header=None)
df_test=pd.read_csv('aksharantar_sampled/hin/hin_test.csv',names=["English","Hindi"],header=None)

maxlength_english=0
maxlength_hindi=0

# Encoder Dictionary Creation
hindi_to_index = {'SOS_token': 0, 'EOS_token': 1, 'PAD_token': 2}
english_to_index = {'SOS_token': 0, 'EOS_token': 1, 'PAD_token': 2}


# Make dictionary for English alphabets
english_alphabets = 'abcdefghijklmnopqrstuvwxyz'
for idx, alphabet in enumerate(english_alphabets):
    english_to_index[alphabet] = idx + 3

# Make dictionary for Hindi characters
hindi_characters = set()
for x in range(len(df)):
    english_word=df.iloc[x]['English']
    hindi_word = df.iloc[x]['Hindi']
    maxlength_english=max(maxlength_english,len(english_word))
    maxlength_hindi=max(maxlength_hindi,len(hindi_word))
    hindi_characters.update(hindi_word) 


for x in range(len(df_test)):
    english_word=df_test.iloc[x]['English']
    hindi_word = df_test.iloc[x]['Hindi']
    maxlength_english=max(maxlength_english,len(english_word))
    maxlength_hindi=max(maxlength_hindi,len(hindi_word))
    hindi_characters.update(hindi_word) 

start = 3
for i, char in enumerate(hindi_characters):
    hindi_to_index[char] = start + i

# Printing the created dictionaries
# print(hindi_to_index)
# print(english_to_index)
maxlength_hindi+=3

# Decoder Dictionary Creation
index_to_hindi = {v: k for k, v in hindi_to_index.items()}
index_to_english = {v: k for k, v in english_to_index.items()}
# print(index_to_english)
# print(index_to_hindi)


#functions to create the encodings required for English and hindi words
def encode_words_english(language,df):
    encoded_words=[]
    maxlength=maxlength_english+1
    to_index=english_to_index
    
    for _, row in df.iterrows():
        language_word = row['English']
        word = torch.zeros(maxlength, dtype=torch.long)+2
        for idx, char in enumerate(language_word):
            word[idx] = to_index[char]
        word[len(language_word)]=to_index['EOS_token']
        encoded_words.append(word)
    encoded_words = torch.stack(encoded_words)
    return encoded_words

def encode_words_hindi(language,df):
    encoded_words=[]
    maxlength=maxlength_hindi
    to_index=hindi_to_index
    
    for _, row in df.iterrows():
        language_word = row['Hindi']
        word = torch.zeros(maxlength, dtype=torch.long)+2
        word[0]=to_index['SOS_token']
        for idx, char in enumerate(language_word):
            word[idx+1] = to_index[char]
        word[len(language_word)]=to_index['EOS_token']
        encoded_words.append(word)
    encoded_words = torch.stack(encoded_words)
    return encoded_words


#contains encoding for training ,validation and testing data.
english_encoded_words=encode_words_english('English',df)
hindi_encoded_words=encode_words_hindi('Hindi',df)
english_encoded_words_val=encode_words_english('English',df_val)
hindi_encoded_words_val=encode_words_hindi('Hindi',df_val)
english_encoded_words_test=encode_words_english('English',df_test)
hindi_encoded_words_test=encode_words_hindi('Hindi',df_test)


#function to reshape the hidden layer 
def reshape_arr(x,num_layers):
    for i in range(1,num_layers,+2):
        if(i==1):tmp=x[i]
        else:tmp+=x[i]
    tmp1=tmp.repeat(num_layers,1,1)
    return tmp1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,embedding_size,num_layers,drop,cell_type,bidirection=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self. num_layers=num_layers
        self.dropout=nn.Dropout(drop)
        self.embedding = nn.Embedding(input_size, embedding_size).to(device)
        self.bidirectional=bidirection
        self.lstm=nn.LSTM(embedding_size,hidden_size,num_layers,dropout=drop,batch_first=False,bidirectional=bidirection).to(device)
        self.rnn = nn.RNN(embedding_size, hidden_size,num_layers,dropout=drop,batch_first=False,bidirectional=bidirection).to(device)
        self.gru = nn.GRU(embedding_size, hidden_size,num_layers,dropout=drop,batch_first=False,bidirectional=bidirection).to(device)
        self.cell_type=cell_type
    def forward(self, input):
        #input:(seq_length,N)
        input=input.T
#         print("einput ",input.shape)
        embedded = self.dropout(self.embedding(input))
#         print("eembed ",embedded.shape)
        #embedded:(seq_length,N,embedding_size)
        if(self.cell_type=="LSTM"):
            output,(hidden,cell)=self.lstm(embedded)
#             print('encodero',output.shape)
#             print('enchid',hidden.shape)
#             print('enccell',cell.shape)
            if(self.bidirectional):
                hidden=reshape_arr(hidden,self.num_layers)
                cell=reshape_arr(cell,self.num_layers)
            return output,(hidden,cell)
            
        if(self.cell_type=="GRU"):
            output, hidden = self.gru(embedded)

        if(self.cell_type=="RNN"):
            output,hidden=self.rnn(embedded)
             
        if(self.bidirectional):
            hidden=reshape_arr(hidden,self.num_layers)
        return  output,hidden
        


class DecoderRNN(nn.Module):
    def __init__(self, input_size,hidden_size, output_size,embedding_size,num_layers,drop,cell_type):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self. num_layers=num_layers
        self.dropout=nn.Dropout(drop)
        self.embedding = nn.Embedding(input_size, embedding_size).to(device)
        self.lstm=nn.LSTM(embedding_size,hidden_size,num_layers,dropout=drop,batch_first=False).to(device)
        self.rnn = nn.RNN(embedding_size, hidden_size,num_layers,dropout=drop,batch_first=False).to(device)
        self.gru = nn.GRU(embedding_size, hidden_size,num_layers,dropout=drop,batch_first=False).to(device)
        self.cell_type=cell_type
        
        self.fc_out = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, input,hidden,cell):
        
        input=input.T
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size,embedding_size]
        
        if(self.cell_type=="RNN"):
            output,hidden = self.rnn(embedded,hidden)
        if(self.cell_type=='GRU'):
            output,hidden = self.gru(embedded,hidden)
        if(self.cell_type=="LSTM"):
            output,(hidden,cell)=self.lstm(embedded,(hidden,cell))
            prediction = self.fc_out(output)
            return prediction,hidden,cell
        #output:[1,batch_size,hidden_size]
        prediction = self.fc_out(output)
        return prediction, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=maxlength_english):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights




class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,cell_type):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.softmax = nn.Softmax(dim=2)
        self.cell_type=cell_type


        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
#         print(source.shape)
#         print(target.shape)
        target_vocab_size = len(hindi_to_index)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)        
        if(self.cell_type=='LSTM'):
            output, (hidden,cell) = self.encoder.forward(source)
        else:
            output, hidden = self.encoder.forward(source)

#         print("output","hidden")
#         print(output,hidden)
        x = target[:,0].reshape(batch_size,1)
        #print(target_len)
        for t in range(1, target_len):
            if(self.cell_type=='LSTM'):
                output, hidden,cell = self.decoder.forward(x, hidden,cell)
            else:
                output, hidden = self.decoder.forward(x, hidden,None)
            
#             print("dout",output.shape)
            outputs[t] = output.squeeze(0)
            teacher_force = random.random() < teacher_forcing_ratio
            output = self.softmax(output)
#             print("doutput ",output.shape)
            top1 = torch.argmax(output,dim = 2)
#             print("top1 ",top1.shape)
            x = target[:,t].reshape(batch_size,1) if teacher_force else top1.T
        return outputs

file1 = open("predictions_vanilla.txt","a")
#file for predictions of the test dataset contains hindi predicted and english predicted.

#to convert the tensors to calculate accuracy
def calculate_predictions(output,target):
    output1=nn.Softmax(dim=2)(output[1:])
    predictions=torch.argmax(output1,dim=2)
    pred=predictions.T
    target1=target[:,1:]
    return pred,target1

#for printing the prediction and target in text file.
def write_to_file(pred,target):
    pred_s=''
    for i in pred:
        if(i in index_to_hindi):
            pred_s+=index_to_hindi[i]
    pred_target=''
    for i in target:
        if(i in index_to_hindi):
            pred_target+=index_to_hindi[i]
    file1.write(pred_s+"        "+pred_target)

#to calculate accuracy 
def calculate_accuracy(model,english_encoded_words,hindi_encoded_words,batch_size,teacher_forcing_ratio):
    correct=0
    total_loss=0
    loss_function=nn.CrossEntropyLoss(reduction='sum')
    
    for i in range(0,len(english_encoded_words),batch_size):
        src=english_encoded_words[i:i+batch_size].to(device)
        target=hindi_encoded_words[i:i+batch_size].to(device)
        output=model.forward(src,target,0)
        pred,target1=calculate_predictions(output,target)
        out = output[1:].reshape(-1, output.shape[2])
        target2 = target[:,1:].T.reshape(-1)
        loss = loss_function(out, target2)
        total_loss += loss.item()
        for t in range(len(pred)):
            if(False):
                write_to_file(pred[t],target1[t])
            if(torch.equal(pred[t],target1[t])):
                correct+=1
    return correct,total_loss





def train(num_layers,enc_dropout,dec_dropout,num_epochs,learning_rate,batch_size,embedding_size,hidden_size,cell_type, wandb_project, wandb_entity):
    wandb.init(
        project=wandb_project,
        entity=wandb_entity
    )
    input_size_encoder=len(english_to_index)
    input_size_decoder=len(hindi_to_index)
    output_size=len(hindi_to_index)
    encoder_net=EncoderRNN(input_size_encoder, hidden_size,embedding_size,num_layers,enc_dropout,cell_type).to(device)
    decoder_net=DecoderRNN(input_size_decoder,hidden_size,output_size,embedding_size,num_layers,dec_dropout,cell_type).to(device)
    model=Seq2Seq(encoder_net,decoder_net,cell_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    correct_predictions=0
    correct_predictions_val=0
    loss_function=nn.CrossEntropyLoss(reduction='sum')
    for epoch in range(num_epochs):
        print(epoch)
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i in range(0,len(english_encoded_words),batch_size):
            src=english_encoded_words[i:i+batch_size].to(device)
            target=hindi_encoded_words[i:i+batch_size].to(device)
            
            output=model(src,target)
            output1=nn.Softmax(dim=2)(output[1:])

            predictions=torch.argmax(output1,dim=2)
   
            out = output[1:].reshape(-1, output.shape[2])
            target1 = target[:,1:].T.reshape(-1)
   
            optimizer.zero_grad()
            loss = loss_function(out, target1)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        #to find the loss and accuracy
        correct_predictions,training_loss=calculate_accuracy(model,english_encoded_words,hindi_encoded_words,batch_size,0)
        correct_predictions_val,val_loss=calculate_accuracy(model,english_encoded_words_val,hindi_encoded_words_val,batch_size,0)
        correct_predictions_test,test_loss=calculate_accuracy(model,english_encoded_words_test,hindi_encoded_words_test,batch_size,0)
        
        Training_loss=total_loss/(len(english_encoded_words)*maxlength_hindi)
        Validation_loss=val_loss/(len(english_encoded_words_val)*maxlength_hindi)
        Validation_accuracy=(correct_predictions_val/len(english_encoded_words_val)*100)
        Test_accuracy=(correct_predictions_test/len(english_encoded_words_test)*100)
        Training_accuracy=(correct_predictions/51200)*100
        print("Training_accuracy:",Training_accuracy)
        print("Validation_accuracy:",Validation_accuracy)
        print("Test_accuracy:",Test_accuracy)
        wandb.log({'Training_accuracy':Training_accuracy,'Epoch':epoch+1,'Training_loss':Training_loss,'Validation_loss':Validation_loss,'Validation_accuracy':Validation_accuracy})
    wandb.run.save()
    wandb.run.finish()

parser = argparse.ArgumentParser(description='calculate accuracy and loss for given hyperparameters')
parser.add_argument('-wp', '--wandb_project', type=str, help='wandb project name', default='Assignment 3')
parser.add_argument('-we', '--wandb_entity', type=str, help='wandb entity', default='cs22m013')
parser.add_argument('-es', '--embedding_size', type=int, help='embedding size', default=256)
parser.add_argument('-nle', '--num_layers', type=int, help='number of layers in encoder', default=3)
parser.add_argument('-hs', '--hidden_size', type=int, help='hidden size', default=512)
parser.add_argument('-bs', '--batch_size', type=int, help='batch size', default=256)
parser.add_argument('-ep', '--num_epochs', type=int, help='epochs', default=20)
parser.add_argument('-ct', '--cell_type', type=str, help='Cell type', default="LSTM")
parser.add_argument('-bdir', '--bidirectional', type=bool, help='bidirectional', default=True)
parser.add_argument('-de', '--enc_dropout', type=float, help='dropout encoder', default=0.4)
parser.add_argument('-dd', '--dec_dropout', type=float, help='dropout decoder', default=0.2)
params = parser.parse_args()
learning_rate = 0.001
if __name__ == '__main__':
    train(params.num_layers,params.enc_dropout,params.dec_dropout,params.num_epochs,learning_rate,params.batch_size,params.embedding_size,params.hidden_size,params.cell_type, params.wandb_project, params.wandb_entity)

# #best config in vanilla.
# num_layers=4
# enc_dropout=0.3
# dec_dropout=0.3
# num_epochs=10
# learning_rate=0.001
# batch_size=512
# hidden_size=1024
# embedding_size=256
# cell_type="LSTM"
