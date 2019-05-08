import numpy as np
import random
import CNN_testdata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels = 1,      # input height
                out_channels = 16,    # n_filters
                kernel_size = 5,      # filter size
                stride = 1,           # filter movement/step
                padding = 2,      
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),   
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(16, 32, 5, 1, 2), 
            nn.ReLU(),  
            nn.MaxPool2d(2), 
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  
        output = self.out(x)
        return output



def get_input(conv):
	testd = CNN_testdata.get_train_data_set()
	test_data = torch.empty((5,1,28,28))
	for i in range(test_data.shape[0]):
		test_data[i]=testd[i].unsqueeze(0)
	output = conv(test_data)
	pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
	set_data = pred_y.tolist()
	return set_data

def get_min():
	global min_index,min_value, iput, out, thrhd, argu, env, I, R, r
	thrhd = 0
	I.append(0)
	
	now_env = []
	for i in range(len(iput)):
		now_env.append(iput[i])
	env.append(now_env)
	count = 0
	# while count < len(iput):
	# 	if iput[count] < min_value:
	# 		min_index = count
	# 		min_value = iput[count]
	for i in range(len(iput)):
		if iput[i] < min_value:
			min_index = i
			min_value = iput[i]

	arg_now = [min_index,min_value]	
	argu.append(arg_now)		
	# print(min_index,min_value,arg_now)
	thrhd = 1
	R.append(thrhd)
	# print('g')
	# print(iput)
	# print(out)
	# print(argu)
	# print(env)
	# print(I)

def append_min():
	global min_index,min_value, iput, out, thrhd, argu, env, I, R, r
	thrhd = 0
	I.append(1)
	arg_now = [min_index,min_value]	
	argu.append(arg_now)
	now_env = []
	for i in range(len(iput)):
		now_env.append(iput[i])
	env.append(now_env)
	out.append(min_value)
	min_value = 6
	iput[min_index] = 9
	thrhd = 1
	R.append(thrhd)
	# print('a')
	# print(iput)
	# print(out)
	# print(argu)
	# print(env)
	# print(I)

def if_finish():
	global min_index,min_value, iput, out, thrhd, argu, env, I, R, r
	thrhd = 0
	I.append(2)
	arg_now = [min_index,min_value]	
	argu.append(arg_now)
	now_env = []
	for i in range(len(iput)):
		now_env.append(iput[i])
	env.append(now_env)
	count = 0
	while count < len(iput):
		if iput[count] == 9:
			count = count + 1
		else:
			break
	if count == len(iput):
		r = 1
	thrhd = 1
	R.append(thrhd)

	# print('f')
	# print(iput)
	# print(out)
	# print(argu)
	# print(env)
	# print(I)


def run():
	global min_index,min_value, iput, out, thrhd, argu, env, I, R, r
	I.append(3)
	arg_now = [min_index,min_value]	
	argu.append(arg_now)
	now_env = []
	for i in range(len(iput)):
		now_env.append(iput[i])
	env.append(now_env)
	thrhd = 0
	get_min()
	append_min()
	if_finish()
	thrhd = r
	R.append(thrhd)

def process():
	global thrhd, argu, env, I
	thrhd = 0
	I.append(4)
	arg_now = [min_index,min_value]	
	argu.append(arg_now)
	now_env = []
	for i in range(len(iput)):
		now_env.append(iput[i])
	env.append(now_env)
	while thrhd == 0:
		run()
		# print(iput)
		# print(out)
		# print(argu)
		# print(env)
		# print(I)
		
	thrhd = 1
	R.append(thrhd)
	# # argu.append([0,0])
	# # I.append(-1)
	# print(R)
	# print(len(env),len(argu),len(I),len(R))

# process()


def train_data():
	global min_index,min_value, iput, out, thrhd, argu, env, I, R, r, BATCH_SIZE, TIME_STEPS, test_iput
	idel_inp = []
	idel_oup = []
	for i in range(BATCH_SIZE):
		# iput = get_input()
		iput = [1,2,3,4,5]
		np.random.shuffle(iput)
		test_iput = iput.copy()
		out = []
		thrhd =0
		I = [0]
		R = []
		argu = [[0,0]]
		env = []
		min_index = 0
		min_value = 6
		r = 0
		iinput = []
		ooput = []
		process()
		# I_now = []
		for i in range(len(I)-1):
			iinput.append([torch.tensor(env[i][0]),torch.tensor(env[i][1]),torch.tensor(env[i][2]),torch.tensor(env[i][3]),torch.tensor(env[i][4]),torch.tensor(argu[i][0]),torch.tensor(argu[i][1]),torch.tensor(I[i])])
		# print(np.shape(I))
		# print(np.shape(argu))
		# print(np.shape(R))
		# print(R)
		# for i in range(len(I)-1):
		# 	if(I[i+1] == 0):                    # encode the program_i to one-hot
		# 		I_now = [1,0,0,0,0]
		# 	if(I[i+1] == 1):
		# 		I_now = [0,1,0,0,0]
		# 	if(I[i+1] == 2):
		# 		I_now = [0,0,1,0,0]
		# 	if(I[i+1] == 3):
		# 		I_now = [0,0,0,1,0]
		# 	else:
		# 		I_now = [0,0,0,0,1]
		# 	ooput.append([torch.tensor(I_now[0]),torch.tensor(I_now[1]),torch.tensor(I_now[2]),torch.tensor(I_now[3]),torch.tensor(I_now[4]),torch.tensor(argu[i+1][0]),torch.tensor(argu[i+1][1]),torch.tensor(R[i])])
		# idel_inp.append(iinput)
		# idel_oup.append(ooput)
		I = torch.tensor(I).reshape(-1,1)
		one_hot = torch.zeros((I.shape[0],5))
		one_hot = one_hot.scatter_(1,I,1.0)
		ooput = []
		p = []
		for i in range(one_hot.shape[0]-1):
			for j in range(one_hot.shape[1]):
				p.append([one_hot[i,j]])
			p.append([torch.tensor(argu[i+1][0]).float()])
			p.append([torch.tensor(argu[i+1][1]).float()])
			p.append([torch.tensor(R[i]).float()])
			ooput.append(p)
			p = []
		idel_inp.append(iinput)
		idel_oup.append(ooput)
	idel_inp = torch.tensor(np.array(idel_inp))
	idel_oup = torch.tensor(np.array(idel_oup))
	return [idel_inp,idel_oup]

def test_data():
	global min_index,min_value, iput, out, thrhd, argu, env, I, R, r, BATCH_SIZE, TIME_STEPS , III
	idel_inp = []
	idel_oup = []

	iput = [1,2,3,4,5]
	np.random.shuffle(iput)

	out = []
	thrhd =0
	I = [0]
	R = []
	argu = [[0,0]]
	env = []
	min_index = 0
	min_value = 6
	r = 0
	iinput = []
	ooput = []
	process()
	I_now = []


	for i in range(len(I)-1):
		iinput.append([torch.tensor(env[i][0]),torch.tensor(env[i][1]),torch.tensor(env[i][2]),torch.tensor(env[i][3]),torch.tensor(env[i][4]),torch.tensor(argu[i][0]),torch.tensor(argu[i][1]),torch.tensor(I[i])])
	I = torch.tensor(I).reshape(-1,1)
	one_hot = torch.zeros((I.shape[0],5))
	one_hot = one_hot.scatter_(1,I,1.0)
	ooput = []
	p = []
	for i in range(one_hot.shape[0]-1):
		for j in range(one_hot.shape[1]):
			p.append([one_hot[i,j]])
		p.append([torch.tensor(argu[i+1][0]).float()])
		p.append([torch.tensor(argu[i+1][1]).float()])
		p.append([torch.tensor(R[i]).float()])
		ooput.append(p)
		p = []
	III = ooput.copy()	
	idel_inp = torch.tensor(np.array(iinput))
	idel_oup = torch.tensor(np.array(ooput))
	return [idel_inp,idel_oup]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(     
            input_size = 8,      
            hidden_size = 64,    
            num_layers = 1,       
            batch_first = True,  
        )

        self.out = nn.Linear(64, 8)   

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  
        out = self.out(r_out)
        return out




if __name__ == '__main__':
	conv = torch.load('CNN.pkl')

	EPOCH = 1        
	BATCH_SIZE = 80
	TIME_STEP = 21     
	INPUT_SIZE = 8    
	LR = 0.001        
	DOWNLOAD_MNIST = True
	iput = get_input(conv)
	np.random.shuffle(iput)
	out = []
	thrhd = 0
	I = [-1]
	R = []
	argu = [[0,6]]
	env = []
	min_index = 0
	min_value = 6
	r = 0
	loss_list = []
	acc_list = []
	rnn = RNN()
	print(rnn)

	optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
	loss_func = nn.MSELoss()   # the target label is not one-hotted

	train_d = train_data()
	# print(type(train_d[0]))
	torch_dataset = Data.TensorDataset(train_d[0].float(), train_d[1].float())

	test_data = test_data()

	# print(type(test_d[0]))
	# test_dataset = Data.TensorDataset(test_d[0].float(), test_d[1].float())

	# 把 dataset 放入 DataLoader
	loader = Data.DataLoader(
	    dataset = torch_dataset,    
	    batch_size = 3,    
	    shuffle = True,               
	    num_workers = 2,            
	)



	# training and testing
	for epoch in range(5):
	    for step, (x, b_y) in enumerate(loader):   # gives batch data
	        b_x = x.view(-1, 21, 8)   # reshape x to (batch, time_step, input_size)
	        b_y=b_y.squeeze()
	        output = rnn(b_x)               # rnn output
	        loss = loss_func(output,b_y)    # cross entropy loss
	        optimizer.zero_grad()           # clear gradients for this training step
	        loss.backward()                 # backpropagation, compute gradients
	        optimizer.step()                # apply gradients
	        if step%10 == 0:
	        	print('step:',step, 'loss:',loss)
	        	loss_list.append(loss)
	        	result_1 = torch.argmax(output[0,:,0:5], dim=1)
	        	result_2 = torch.argmax(b_y[0,:,0:5], dim=1)
 
	        	acc=torch.sum((result_1==result_2).float())/result_1.shape[0]
	        	# output=output.clone().detach().numpy()
	        	# b_y=b_y.clone().detach().numpy()
	        	# ac=np.sum(output==b_y)/output.shape[0]
	        	err = 1-acc
	        	print('         error:', err)
	        	acc_list.append(err)

	# loss_list.save('lr0.001.npy')
	# test_dt = test_data[0].unsqueeze(0)     
	# test_res = rnn(test_dt.float())
	# result = torch.argmax(test_res[0,:,0:5], dim=1)
	# print(result)
	acc_list=np.array(acc_list)
	loss_list=np.array(loss_list)
	np.save('ta_1.npy',acc_list)
	np.save('ls_1.npy',loss_list)
	# np.save('ta_1_3.npy',acc_list)
	# np.save('ls_1_3.npy',loss_list)


