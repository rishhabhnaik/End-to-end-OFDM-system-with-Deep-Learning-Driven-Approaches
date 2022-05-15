from __future__ import division
import numpy as np
import scipy.interpolate 
# import tensorflow as tf
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 64
CP = K//4
P = 64 # number of pilot carriers per OFDM block
#pilotValue = 1+1j
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
#pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
#P = P+1
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 4
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

payloadBits_per_OFDM = K*mu

# SNRdb = 12  # signal to noise-ratio in dB at the receiver 
SNRdb_list = [5, 10, 15, 20, 25]

Clipping_Flag = False 
#Clipping_Flag = False

mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

demapping_table = {v : k for k, v in mapping_table.items()}


def map_bits(bits):
    key = tuple(bits.tolist())
    return mapping_table[key]

def Clipping (x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL*sigma
    x_clipped = x  
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))

    return x_clipped

def PAPR (x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB

def Modulation(bits):                                        
    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    # return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation
    return np.apply_along_axis(map_bits, axis=1, arr=bit_r)                                   # This is just for QAM modulation

def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    #symbol = np.zeros(K) 
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = Data  # allocate the pilot subcarriers
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))

def ofdm_simulate(codeword, channelResponse,SNRdb):   
    
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX,CR)                            # add clipping 
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword,CR) # add clipping 
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    #OFDM_RX_noCP_codeword = DFT(OFDM_RX_noCP_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 

def ofdm_simulate_single_without_CP(codeword, channelResponse):  
    
    codeword_qam = Modulation(codeword)
    OFDM_data_codeword = OFDM_symbol(codeword_qam)
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)

    # using a new ofdm symbol for the prefix
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    codeword_noise = Modulation(codeword)
    OFDM_data_nosie = OFDM_symbol(codeword_noise)
    OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)   
    cp = OFDM_time_noise[-CP:]               # take the last CP samples ...
    OFDM_withCP_cordword = np.hstack([cp,OFDM_time_codeword])
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    
    #return np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))) , abs(channelResponse) #sparse_mask
    return np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))), abs(channelResponse)


Pilot_file_name = 'Pilot_'+str(P)
if os.path.isfile(Pilot_file_name):
    print ('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')


pilotValue = Modulation(bits)
CR = 1 

n_hidden_1 = 500
n_hidden_2 = 250 # 1st layer num features
n_hidden_3 = 120 # 2nd layer num features
n_input = 256  
n_output = 16 # every 16 bit are predicted by a model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(torch.nn.Module):
    def __init__(self):
      super(Encoder, self).__init__()       
     
      self.fc1 = torch.nn.Linear(n_input, n_hidden_1)     
      self.fc2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
      self.fc3 = torch.nn.Linear(n_hidden_2, n_hidden_3)
      self.fc4 = torch.nn.Linear(n_hidden_3, n_output)


    def forward(self, x):
      # Encoder Hidden layer with sigmoid activation #1

      layer_1 = F.relu(self.fc1(x))
      layer_2 = F.relu(self.fc2(layer_1))
      layer_3 = F.relu(self.fc3(layer_2))
      layer_4 = F.sigmoid(self.fc4(layer_3))
      return layer_4

def training(SNRdb):     
        # Training parameters
        training_epochs = 20
        batch_size = 256
        display_step = 5
        test_step = 1000
        examples_to_show = 10   
        # Network Parameters

        encoder = Encoder()
        encoder.to(device)
        # Targets (Labels) are the input data.
        # print(list(encoder.parameters())) # For debug purpose remove later!!
        # Define loss and optimizer, minimize the squared error
        criterion = torch.nn.MSELoss(reduction='mean')
        criterion2 = torch.nn.L1Loss()
        optimizer = torch.optim.RMSprop(encoder.parameters(), lr=1e-3) # Check default RMSProp parameters later if not working

        # The H information set
        # H_folder_train = '/content/drive/Shareddrives/EECS 555/Colab notebooks/EECS555/Data for OFDM_DNN/H_dataset/'
        # H_folder_test = '/content/drive/Shareddrives/EECS 555/Colab notebooks/EECS555/Data for OFDM_DNN/H_dataset/'
#        H_folder_train = '/content/drive/Shareddrives/EECS 555/Colab notebooks/EECS555/Data for OFDM_DNN/Synthetic_dataset/'
#        H_folder_test = '/content/drive/Shareddrives/EECS 555/Colab notebooks/EECS555/Data for OFDM_DNN/Synthetic_dataset/'

### For running on local machine only
        H_folder_train = sys.argv[1]
        H_folder_test = sys.argv[1]
### For running on local machine only

        train_idx_low = 1
        train_idx_high = 301
        test_idx_low = 301
        test_idx_high = 401
        # Saving Channel conditions to a large matrix
        channel_response_set_train = []
        for train_idx in range(train_idx_low,train_idx_high):
            print("Processing the ", train_idx, "th document")
            H_file = H_folder_train + str(train_idx) + '.txt'
            with open(H_file) as f:
                for line in f:
                    try:
                      numbers_str = line.split()
                      numbers_float = [float(x) for x in numbers_str]
                      h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)]) + 1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                      channel_response_set_train.append(h_response)
                    except ValueError as V:
                      continue
        
        channel_response_set_test = []
        for test_idx in range(test_idx_low,test_idx_high):
            print("Processing the ", test_idx, "th document")
            H_file = H_folder_test + str(test_idx) + '.txt'
            with open(H_file) as f:
                for line in f:
                    try:
                      numbers_str = line.split()
                      numbers_float = [float(x) for x in numbers_str]
                      h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                      channel_response_set_test.append(h_response)
                    except ValueError as V:
                      continue

        print ('length of training channel response', len(channel_response_set_train), 'length of testing channel response', len(channel_response_set_test))
        training_epochs = 250
        learning_rate_current = 0.001
        for epoch in range(training_epochs):
            print(epoch)
            if epoch > 0 and epoch%100 ==0:
                learning_rate_current = learning_rate_current / 5                    

            avg_cost = 0.
            total_batch = 50 
            for g in optimizer.param_groups: 
                g['lr'] = learning_rate_current     # Changing the learning rate with epochs
 
            for index_m in range(total_batch):
                input_samples = []
                input_labels = []
                for index_k in range(0, 1000):
                    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
                    channel_response = channel_response_set_train[np.random.randint(0,len(channel_response_set_train))]
                    try:
                      signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)   
                    except ValueError as V:
                      continue
                    input_labels.append(bits[16:32])
                    input_samples.append(signal_output)  
                batch_x = np.asarray(input_samples)
                batch_y = np.asarray(input_labels)            

                y_pred = encoder(torch.from_numpy(batch_x).float().to(device))
                loss = criterion(y_pred, torch.from_numpy(batch_y.astype(np.float32)).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                c = loss.item()

                avg_cost += c / total_batch

            if epoch % display_step == 0:
                print("Epoch:",'%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
                input_samples_test = []
                input_labels_test = []
                test_number = 1000
                # set test channel response for this epoch                    
                if epoch % test_step == 0:
                    print ("Big Test Set ")
                    test_number = 10000
                for i in range(0, test_number):
                    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, )) 
                    try:                       
                      channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                      signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)
                    except ValueError as V:
                      continue
                    input_labels_test.append(bits[16:32])
                    input_samples_test.append(signal_output)
                batch_x = np.asarray(input_samples_test)
                batch_y = np.asarray(input_labels_test)
                y_pred = encoder(torch.from_numpy(batch_x).float().detach().to(device)).to(device)
                loss1_L1 = criterion2(y_pred, torch.from_numpy(batch_y.astype(np.float32)).to(device))
                mean_error = loss1_L1.item()
                # mean_error = torch.mean(abs(y_pred - torch.from_numpy(batch_y).detach()), keepdim=True)
                # mean_error_rate = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))               
                # mean_error_rate = 1 - torch.mean(torch.mean(), keepdim=True)
                mean_error_rate = 1 - np.mean(np.mean(np.equal(np.sign(y_pred.detach().cpu().numpy()-0.5), np.sign(batch_y-0.5)),axis=1))
                print("OFDM Detection QAM output number is", n_output, ",SNR = ", SNRdb, ",Num Pilot = ", P,", prediction and the mean error on test set are:", mean_error, mean_error_rate)

                batch_x = np.asarray(input_samples)
                batch_y = np.asarray(input_labels)
                y_pred = encoder(torch.from_numpy(batch_x).float().detach().to(device)).to(device)
                # mean_error = torch.mean(abs(y_pred - batch_y), keepdim=True)
                loss2_L1 = criterion2(y_pred, torch.from_numpy(batch_y.astype(np.float32)).to(device))
                mean_error = loss2_L1.item()

                # mean_error_rate = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))
                mean_error_rate = 1 - np.mean(np.mean(np.equal(np.sign(y_pred.detach().cpu().numpy()-0.5), np.sign(batch_y-0.5)),axis=1))
                print("Prediction and the mean error on train set are:", mean_error, mean_error_rate)



        print("optimization finished")
        return encoder, mean_error_rate


encoder_list = []
ber_list = []
for SNRdb in SNRdb_list:
  encoder, mean_error_rate = training(SNRdb)
  encoder_list.append(encoder)
  ber_list.append(mean_error_rate)

# !!! Uncomment this if running on local machine !!!
import pickle
import time
with open("./encoder_file_{0}.pkl".format(int(time.time())), "wb") as encFile:
  pickle.dump(encoder_list, encFile)

with open("./ber_list_file_{0}.pkl".format(int(time.time())), "wb") as berFile:
  pickle.dump(ber_list, berFile)

