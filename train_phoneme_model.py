import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import numpy as np
from phoneme_model import LSTM_custom,MOSI_phonemenal_classifier
import gzip
import cPickle as pkl
import matplotlib.pyplot as plt
from mosi_helper import *
from mosi_data_util import *
from mosi_model_evaluator import MosiEvaluator
import datetime
import csv 
import sys

model_version="../experiment/phoneme_model/data_loader/"
#model_version="/scratch/mhasan8/experiment/phoneme_model/data_loader/"

def save_result(model_name,eval_results,params):
	print params

	eval_results=[model_name]+eval_results

	for key,value in params.iteritems():
		eval_results.append(value)

	result_csv_file = model_version+"results/mab_results.csv"	
	with open(result_csv_file, 'a') as out_f:
		wr = csv.writer(out_f)
		wr.writerow(eval_results)
	out_f.close()

def print_loss(e_tr_losses,e_val_losses,model_name):
	fig_name=model_version+"fig/"+model_name+".png"
	legend=["train_loss","val_loss"]
	plt.plot(e_tr_losses)
	plt.plot(e_val_losses)
	plt.ylabel('Loss')
	plt.xlabel('iteration')
	plt.legend(legend, loc='upper right')
	title="Loss plot for "+model_name
	plt.title(title)
	plt.savefig(fig_name)
	plt.close()

train_x,train_y=load_data('../mosi_data/normalized/phoneme_based/COVAREP/train_matrix.pkl')
print "len train",len(train_y)
train_data_loader=get_data_loader(train_x,train_y)
print("loaded train data loader")
test_x,test_y=load_data('../mosi_data/normalized/phoneme_based/COVAREP/test_matrix.pkl')
print("loaded test")
print "len test",len(test_y)
valid_x,valid_y=load_data('../mosi_data/normalized/phoneme_based/COVAREP/valid_matrix.pkl')
print "len valid",len(valid_y)
valid_data_loader=get_data_loader(valid_x,valid_y)
print("loaded valid")

# train_x,train_y=load_data('../mosi_data/normalized/phoneme_based/COVAREP/valid_matrix.pkl')

# print("loaded train data loader")
# test_x,test_y=train_x[20:35],train_y[20:35]

# valid_x,valid_y=train_x[27:35],train_y[27:35]
# print("loaded valid")

# train_x,train_y=train_x[20:40],train_y[20:40]
# train_data_loader=get_data_loader(train_x,train_y)
# valid_data_loader=get_data_loader(valid_x,valid_y)


def train_epoch(mosi_model,opt,criterion):
	losses=[]
	mosi_model.train()
	for i, data in enumerate(train_data_loader):
		seq ,label = data
		opt.zero_grad()
		mini_batch_losses=[]
		for j,x in enumerate(seq):
			x=get_unpad_data(x)
			y=variablize(torch.FloatTensor([label[j]]))
			y_hat=mosi_model.forward(x)
			loss = criterion(y_hat, y)
			mini_batch_losses.append(loss)

		mini_batch_loss=reduce(torch.add,mini_batch_losses)/len(mini_batch_losses)
		mini_batch_loss.backward()
		opt.step()
		losses.append(mini_batch_loss.cpu().data.numpy())

	return np.nanmean(losses)

def validation_loss(mosi_model,criterion):

	mosi_model.eval()
	losses = []
	with torch.no_grad():
		for i, data in enumerate(valid_data_loader):
			seq ,label = data
			mini_batch_losses=[]
			for j,x in enumerate(seq):
				x=get_unpad_data(x)
				y=variablize(torch.FloatTensor([label[j]]))
				y_hat=mosi_model.forward(x)
				loss = criterion(y_hat, y)
				mini_batch_losses.append(loss)

			mini_batch_loss=reduce(torch.add,mini_batch_losses)/len(mini_batch_losses)
			losses.append(mini_batch_loss.cpu().data.numpy())
		# for i in range(len(valid_x)):
		# 	x=valid_x[i]
		# 	y=variablize(torch.FloatTensor([valid_y[i]]))
		# 	y_hat=mosi_model.forward(x)
		# 	loss = criterion(y_hat, y)
		# 	losses.append(loss.cpu().data.numpy())

	return np.nanmean(losses)


def evaluate_best_model(model_name,params,epoch_num):

	evaluator=MosiEvaluator()
	model_file=model_version+"models/"+model_name

	d_lan_param={'input_dim':1,'hidden_dim':1}
	d_audio_param={'input_dim':1,'hidden_dim':1}
	d_face_param={'input_dim':1,'hidden_dim':1}
	
	best_model=MOSI_phonemenal_classifier(d_lan_param,d_audio_param,d_face_param,1,0.1)
	best_model.load(open(model_file,'rb'))

	comment="validtion evaluation for best model: "+model_name
	print(comment)
	eval_val = evaluator.evaluate(best_model,valid_x,valid_y)
	comment="test evaluation for best model: "+model_name
	print(comment)
	eval_test = evaluator.evaluate(best_model,test_x,test_y)

	eval_results=eval_val+eval_test+[epoch_num]
	save_result(model_name,eval_results,params)


def evaluate_new_valid_model(best_model,model_name,params,epoch_num):
	evaluator=MosiEvaluator()
	comment="validtion evaluation for best model: "+model_name
	print(comment)
	eval_val = evaluator.evaluate(best_model,valid_x,valid_y)
	comment="test evaluation for best model: "+model_name
	print(comment)
	eval_test = evaluator.evaluate(best_model,test_x,test_y)

	eval_results=eval_val+eval_test+[epoch_num]
	save_result(model_name,eval_results,params)


def train_mosi_sentiments(mosi_model,params):

	evaluator=MosiEvaluator()

	model_name="phoneme_"+str(params)
	model_file=model_version+"models/"+model_name

	opt = optim.Adam(mosi_model.parameters(), lr=params['lr'])
	criterion = nn.BCEWithLogitsLoss()
	e_tr_losses = []
	e_val_losses = []
	num_epochs = 400

	best_valid_loss=np.inf

	for e in range(num_epochs):
		train_loss=train_epoch(mosi_model, opt, criterion)
		e_tr_losses.append(train_loss)

		valid_loss=validation_loss(mosi_model,criterion)
		e_val_losses.append(valid_loss)

		if valid_loss<best_valid_loss:
			best_valid_loss=valid_loss
			print "best valid loss",best_valid_loss	
			evaluate_new_valid_model(mosi_model,model_name,params,e)
			try:		
				mosi_model.save(open(model_file,'wb'))				
			except:
				print "error happended due to save model file",model_name,"epoch num",e
				pass	

		if (e%5==0):
			print_loss(e_tr_losses,e_val_losses,model_name)

		print "epoch",e

	evaluate_best_model(model_name,params,num_epochs)


if __name__=='__main__':

	print("started")
	start_time = time.time()	
	s_i,e_i=int(sys.argv[1]),int(sys.argv[2])

	fp=gzip.open("params_set.pkl",'rb')
	params_list=pkl.load(fp)
	params_list=params_list[s_i:e_i]

	out_dim=1
	#params_list=[(40,36,20,0.0001,0.05)]
	for param in params_list:
		print param 
		(lan_hid_dim,audio_hid_dim,face_hid_dim,learning_rate,drop_out)=param 
		lan_param={'input_dim':len(w_dim_index),'hidden_dim':lan_hid_dim}
		audio_param={'input_dim':len(covarep_dim_index),'hidden_dim':audio_hid_dim}
		face_param={'input_dim':len(facet_dim_index),'hidden_dim':face_hid_dim}

		print lan_param,audio_param,face_param

		if (helper_gpu_mode and torch.cuda.is_available()):
			print("gpu found")
			try:
				mosi_model=MOSI_phonemenal_classifier(lan_param,audio_param,face_param,out_dim,drop_out).cuda()
			except:
				print "error happended in model",param 
				pass
		else:
			try:
				mosi_model=MOSI_phonemenal_classifier(lan_param,audio_param,face_param,out_dim,drop_out)
			except:
				print "error happended in model",param 
				pass
		params_config={"l":lan_hid_dim,"a":audio_hid_dim,"f":face_hid_dim,"lr":learning_rate,"dr":drop_out}
		print params_config
		train_mosi_sentiments(mosi_model,params_config)


	time_str="multiple attention full data program run time "+str((time.time() - start_time))+"seconds ---"
	f_name=model_version+"out.txt"
	f=open(f_name,"a")
	f.write(time_str)
	#print("---program run time  %s seconds ---" % (time.time() - start_time))
