from torchtext.data import Field,TabularDataset,BucketIterator
import torch.nn as nn
import spacy
import torch
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm 



spacy_en  =  spacy.load("en_core_web_sm")



print("Modules Loaded")


def tokenize(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]


text = Field(tokenize=tokenize,sequential=True,lower=True,use_vocab=True)
target = Field(sequential=False,use_vocab=False)


fields = {"text":("t",text),"target":("s",target)} # t->text , s->target/score


train_data,test_data = TabularDataset.splits(path="data/sms_spam",train="train.csv",test="test.csv",format="csv",fields=fields)


text.build_vocab(train_data,max_size=10000,min_freq=1)\

train_iterator ,test_iterator = BucketIterator.splits((train_data,test_data),batch_size=2)



class Model(nn.Module):
	def __init__(self,vocab_size,embed_dim):
		super(Model,self).__init__()


		self.embedding = nn.Embedding(vocab_size,embed_dim)
		self.lstm = nn.LSTM(embed_dim,256,num_layers=2)
		self.fc = nn.Linear(256*2,1)
		self.dropout = nn.Dropout(0.2)
		self.sig = nn.Sigmoid()


	def forward(self,x):
		x=  self.dropout(self.embedding(x))
		output,(h,c) =  self.lstm(x)
		h = torch.cat([h[-2],h[-1]],dim=1)
		h = self.dropout(h)
		out = self.fc(h)
		out = self.sig(out)
		return out 


vocab_size = len(text.vocab)
embed_dim = 100 
model = Model(vocab_size,embed_dim)
lr=0.001
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss = nn.BCELoss()
epochs =  5




losses = []


print("Training Progress...")

for i in range(epochs):

	print("--RUNNING EPOCH : {}".format(i))

	batch_losses = []
	for step,batch in enumerate(train_iterator):


		preds = model(batch.t).squeeze(1)
		loss_ = loss(preds,batch.s.float())
		batch_losses.append(loss_.item())
		
		loss_.backward()

		optimizer.step()
		optimizer.zero_grad()

		if step % 100 == 0:
			print("STEP : {} LOSS :{}".format(step,loss_.item()))

	losses.append(np.mean(batch_losses))




plt.plot(range(epochs),losses)
plt.show()




def accuracy(preds,y):
    rounded = torch.round(preds)
    correct = (rounded == y).float()
    return correct.sum()/ len(correct)


epoch_acc = 0
total_steps =  len(test_iterator)

model.eval()
with torch.no_grad():
    for batch in test_iterator:
        pred = model(batch.t).squeeze()
        acc = accuracy(pred,batch.s)
        epoch_acc +=  acc 
        
       
       
print("Accuracy : {}".format(epoch_acc/total_steps))







