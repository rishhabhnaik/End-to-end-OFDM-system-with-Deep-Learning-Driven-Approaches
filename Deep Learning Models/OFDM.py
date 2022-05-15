import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate
from scipy.io import savemat
import os
from tqdm import tqdm

class OFDM:
    def __init__(self, no_of_OFDM_subcarriers, no_of_OFDM_symbols_per_frame, SNRdb) -> None:

        self.K = no_of_OFDM_subcarriers # number of OFDM subcarriers
        self.CP = self.K//4  # length of the cyclic prefix: 25% of the block

        
        self.P = 8 # number of pilot carriers per OFDM block
        self.pilotValue = 3+3j # The known value each pilot transmits

        self.allCarriers = np.arange(self.K)  # indices of all subcarriers ([0, 1, ... K-1])

        self.pilotCarriers = self.allCarriers[::self.K//self.P] # Pilots is every (K/P)th carrier.

        # For convenience of channel estimation, let's make the last carriers also be a pilot
        self.pilotCarriers = np.hstack([self.pilotCarriers, np.array([self.allCarriers[-1]])])
        self.P = self.P+1

        # data carriers are all remaining carriers
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)

        self.QAM_string = "16_QAM"
        self.mu = 4 # bits per symbol (i.e. 16QAM)
        self.payloadBits_per_OFDM = len(self.dataCarriers)*self.mu  # number of payload bits per OFDM symbol

        self.set_QAM_mapping_table()

        self.demapping_table = {v : k for k, v in self.mapping_table.items()}

        # self.channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
        # self.channelResponse = np.array([1])  # the impulse response of the wireless channel

        # self.H_exact = np.fft.fft(self.channelResponse, self.K)
        # plt.plot(self.allCarriers, abs(self.H_exact))

        self.SNRdb = SNRdb  # signal to noise-ratio in dB at the receiver 

        self.ofdmSymbolsPerFrame = no_of_OFDM_symbols_per_frame

        self.no_taps=5
        self.H_exact=np.zeros((self.ofdmSymbolsPerFrame,self.K),dtype=np.complex64)
        self.channelResponse=np.zeros((self.ofdmSymbolsPerFrame,self.no_taps),dtype=np.complex64)
        self.center_freq = 5.16

        self.bits_stream = np.zeros((self.ofdmSymbolsPerFrame, self.payloadBits_per_OFDM))
        self.OFDM_Transmitted_frames = np.zeros((self.ofdmSymbolsPerFrame, self.K+self.CP), dtype =np.complex64)
        self.OFDM_Channel_frames = np.zeros((self.ofdmSymbolsPerFrame, self.K+self.CP), dtype =np.complex64)
        self.OFDM_receiver_Demod = np.zeros((self.ofdmSymbolsPerFrame, self.K), dtype=np.complex64)

        # Variables for MMSE
        self.beta = 17/9 # For 16 QAM

    def set_QAM_mapping_table(self):
        if self.QAM_string == "16_QAM":
            self.mapping_table = {
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
        elif self.QAM_string == "4_QAM":
            self.mapping_table = {
                (0,0) : -1-1j,
                (0,1) : -1+1j,
                (1,0) : +1-1j,
                (1,1) : +1+1j
            }

    def OFDM_Frame_Transmitter(self):
        for i in range(self.ofdmSymbolsPerFrame):
            bits = np.random.binomial(n=1, p=0.5, size=(self.payloadBits_per_OFDM, ))
            self.bits_stream[i, :] = bits
            self.OFDM_Transmitted_frames[i, :] = self.OFDM_transmitter(bits)
            

        return self.OFDM_Transmitted_frames
    
    def OFDM_Frame_Receiver(self, OFDM_RX_frame):
        OFDM_RX_matrix = OFDM_RX_frame.reshape((self.ofdmSymbolsPerFrame, self.K+self.CP))
        H_est_matrix = np.zeros(( self.ofdmSymbolsPerFrame, self.K), dtype=np.complex64)
        for i,OFDM_sym in enumerate(OFDM_RX_matrix):
            self.OFDM_receiver_Demod[i,:] = self.OFDM_receiver(OFDM_sym)

            # Hest = self.channelEstimate(self.OFDM_receiver_Demod[i,:])
            Hest = self.MMSE_channelEstimate(self.OFDM_receiver_Demod[i,:], i)
            H_est_matrix[i,:] = Hest
        
        return H_est_matrix, self.OFDM_receiver_Demod
         
    def OFDM_transmitter(self, bits):
        # Randomly generated bits
        self.bits = bits

        # Serial To Parallel Converter which groups data into bits per Symbol
        def SerialToParallel(bits):
            return bits.reshape((len(self.dataCarriers), self.mu))
        self.bits_SP = SerialToParallel(self.bits)

        # Map the bits to QAM complex mapping.
        def Mapping(bits):
            return np.array([self.mapping_table[tuple(b)] for b in bits])
        QAM = Mapping(self.bits_SP)

        # Prepare One OFDM Symbol.
        def OFDM_symbol(QAM_payload):
            symbol = np.zeros(self.K, dtype=complex) # the overall K subcarriers
            symbol[self.pilotCarriers] = self.pilotValue  # allocate the pilot subcarriers 
            symbol[self.dataCarriers] = QAM_payload  # allocate the pilot subcarriers
            return symbol
        self.OFDM_data = OFDM_symbol(QAM)
        # print ("Number of OFDM carriers in frequency domain: ", len(self.OFDM_data))

        # Compute IFFT to convert frequency domain to Time domain
        def IDFT(OFDM_data):
            return np.fft.ifft(OFDM_data)
        self.OFDM_time = IDFT(self.OFDM_data)
        # print ("Number of OFDM samples in time-domain before CP: ", len(self.OFDM_time))

        # Add Cyclic Prefix bits to the Received Time signals.
        def addCyclicPrefix(OFDM_time):
            cp = OFDM_time[-self.CP:]               # take the last CP samples ...
            return np.hstack([cp, OFDM_time])  # ... and add them to the beginning
        self.OFDM_withCP = addCyclicPrefix(self.OFDM_time)
        # print ("Number of OFDM samples in time domain with CP: ", len(self.OFDM_withCP))
        return self.OFDM_withCP

    def generate_random_taps(self, nsymbol):  
        for i in range(self.no_taps):
            list_rx = [np.random.uniform(0,1) for i in range(self.no_taps-1)]
            list_rx=np.sort(list_rx)

            tou= np.hstack((0,list_rx)) 

            ray= np.hstack((1,np.random.rayleigh(0.5, self.no_taps-1)))
            self.channelResponse[nsymbol, i] = ray[i]*np.exp(-1j*2*self.center_freq*tou[i])
        self.H_exact[nsymbol, :] = np.fft.fft(self.channelResponse[nsymbol,:], self.K)

    def Channel(self, OFDM_TX):
        for i in range(self.ofdmSymbolsPerFrame):
            self.generate_random_taps(i)
            convolved = np.convolve(OFDM_TX[i,:], self.channelResponse[i,:], mode='full')
            convolved = convolved[:(self.K+self.CP)]
            # Generate complex noise with given variance
            signal_power = np.mean(abs(convolved**2))
            sigma2 = signal_power * 10**(-self.SNRdb/10)  # calculate noise power based on signal power and SNR
            # print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
            noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
            self.OFDM_Channel_frames[i,:] = convolved + noise
        return np.hstack(self.OFDM_Channel_frames)

    def OFDM_receiver(self, OFDM_RX_CP):

        # Remove Cyclic Prefix bits from the Received Time signals.
        def removeCP(signal):
            return signal[self.CP:(self.CP+self.K)]
        self.OFDM_RX_noCP = removeCP(OFDM_RX_CP)

        # Demodulation from Time to Frequency using FFT
        def DFT(OFDM_RX):
            return np.fft.fft(OFDM_RX)
        self.OFDM_demod = DFT(self.OFDM_RX_noCP)

        return self.OFDM_demod

    def channelEstimate(self, OFDM_demod):
        pilots = OFDM_demod[self.pilotCarriers]  # extract the pilot values from the RX signal
        Hest_at_pilots = pilots / self.pilotValue # divide by the transmitted pilot values
        
        # Perform interpolation between the pilot carriers to get an estimate
        # of the channel in the data carriers. Here, we interpolate absolute value and phase 
        # separately
        Hest_abs = scipy.interpolate.interp1d(self.pilotCarriers, abs(Hest_at_pilots), kind='linear')(self.allCarriers)
        Hest_phase = scipy.interpolate.interp1d(self.pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(self.allCarriers)
        Hest = Hest_abs * np.exp(1j*Hest_phase)
        
        # plt.plot(self.allCarriers, abs(self.H_exact), label='Correct Channel')
        # plt.stem(self.pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
        # plt.plot(self.allCarriers, abs(Hest), label='Estimated channel via interpolation')
        # plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
        # plt.ylim(0,2)
        
        return Hest

    def MMSE_channelEstimate(self, OFDM_demod, nsymbol):
        # import pdb;pdb.set_trace()
        C_response = np.array(self.H_exact[nsymbol,:]).reshape(-1,1)
        C_response_H = np.conj(C_response).T
        R_HH = np.matmul(C_response, C_response_H)
        snr = 10**(self.SNRdb/10)
        # W = R_HH/(R_HH+(self.beta/snr)*np.eye(self.K))
        W = np.matmul(R_HH,np.linalg.inv((R_HH+(self.beta/snr)*np.eye(self.K))))
        HhatLS = self.channelEstimate(OFDM_demod)
        HhatLS = HhatLS.reshape(-1,1)
        HhatLMMSE = np.matmul(W, HhatLS)

        return HhatLMMSE.squeeze()

    def OFDM_equalize(self, OFDM_demod, Hest):
        return OFDM_demod / Hest
    
    def get_payload_stream(self, equalized):
        return equalized[:, self.dataCarriers]

    def get_payload(self, equalized):
        return equalized[self.dataCarriers]

    def Demapping_payload_stream(self, QAM):
        demapped_bit_stream = np.zeros((self.ofdmSymbolsPerFrame, len(self.dataCarriers), self.mu))
        hardDecision_stream = np.zeros((self.ofdmSymbolsPerFrame, self.dataCarriers.shape[0]), dtype =np.complex64)
        for i, QAM_OFDM_symbol in enumerate(QAM):
            demapped_bits, hardDecision = self.Demapping(QAM_OFDM_symbol)
            demapped_bit_stream[i,:,:] = demapped_bits
            hardDecision_stream[i,:] = hardDecision
        return demapped_bit_stream, hardDecision_stream

    def Demapping(self, QAM):
        # array of possible constellation points
        constellation = np.array([x for x in self.demapping_table.keys()])
        
        # calculate distance of each RX point to each possible point
        dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
        
        # for each element in QAM, choose the index in constellation 
        # that belongs to the nearest constellation point
        const_index = dists.argmin(axis=1)
        
        # get back the real constellation point
        hardDecision = constellation[const_index]
        
        # transform the constellation point into the bit groups
        return np.vstack([self.demapping_table[C] for C in hardDecision]), hardDecision

    def get_bit_stream(self):
        return self.bits_stream

    def display_carrier_distribution(self):
        print ("allCarriers:   %s" % self.allCarriers)
        print ("pilotCarriers: %s" % self.pilotCarriers)
        print ("dataCarriers:  %s" % self.dataCarriers)
        plt.plot(self.pilotCarriers, np.zeros_like(self.pilotCarriers), 'bo', label='pilot')
        plt.plot(self.dataCarriers, np.zeros_like(self.dataCarriers), 'ro', label='data')

    def display_constellation_map(self):
        for b3 in [0, 1]:
            for b2 in [0, 1]:
                for b1 in [0, 1]:
                    for b0 in [0, 1]:
                        B = (b3, b2, b1, b0)
                        Q = self.mapping_table[B]
                        plt.plot(Q.real, Q.imag, 'bo')
                        plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')


def main():
    
    # no_of_epochs = 100000
    SNRdb = 1
    no_of_OFDM_subcarriers = 64
    no_of_OFDM_symbols_per_frame = 14
    no_of_epochs = 100000
    
    if not os.path.isdir(f"data\Dataset_for_{SNRdb}db"):
        os.mkdir(f"data\Dataset_for_{SNRdb}db")        


    print(f"Creating Dataset for SNR: {SNRdb} dB")
    perfect_dataset = np.zeros((no_of_epochs, no_of_OFDM_subcarriers, no_of_OFDM_symbols_per_frame), dtype=np.complex64)
    noisy_dataset = np.zeros((no_of_epochs, no_of_OFDM_subcarriers, no_of_OFDM_symbols_per_frame), dtype=np.complex64)
    for epoch in tqdm(range(no_of_epochs)):
        OFDM_system = OFDM(no_of_OFDM_subcarriers, no_of_OFDM_symbols_per_frame, SNRdb)
        # print(f"SNR set is: {OFDM_system.SNRdb} dB")
        # bits = np.random.binomial(n=1, p=0.5, size=(OFDM_system.payloadBits_per_OFDM, ))
        # OFDM_TX = OFDM_system.OFDM_transmitter(bits)
        
        OFDM_TX = OFDM_system.OFDM_Frame_Transmitter()

        OFDM_RX = OFDM_system.Channel(OFDM_TX)

        # plt.figure(figsize=(8,2))
        # plt.plot(abs(OFDM_TX), label='TX signal')
        # plt.plot(abs(OFDM_RX), label='RX signal')
        # plt.legend(fontsize=10)
        # plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
        # plt.grid(True)
        # plt.show()

        H_est_matrix, OFDM_Frame_demod = OFDM_system.OFDM_Frame_Receiver(OFDM_RX)

        equalized_Hest = OFDM_system.OFDM_equalize(OFDM_Frame_demod, H_est_matrix)
        
        QAM_est = OFDM_system.get_payload_stream(equalized_Hest)

        PS_est, hardDecision = OFDM_system.Demapping_payload_stream(QAM_est)

        # def ParallelToSerial(bits):
        #     return bits.reshape((-1,))
        # bits_est = ParallelToSerial(PS_est)

        # bits_transmitted = OFDM_system.get_bit_stream()
        # bits_transmitted = bits_transmitted.reshape((-1,))

        perfect_dataset[epoch,:,:] = OFDM_system.H_exact.T
        noisy_dataset[epoch,:,:] = OFDM_system.OFDM_receiver_Demod.T
        ######print ("Obtained Bit error rate: ", np.sum(abs(bits_transmitted-bits_est))/len(bits_transmitted))
        # for qam, hard in zip(QAM_est, hardDecision):
        #     plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
        #     plt.plot(hardDecision.real, hardDecision.imag, 'ro')


        # plt.plot(QAM_est.real, QAM_est.imag, 'bo')

    savemat(f"data/Dataset_for_{SNRdb}db/Perfect_H_dataset_for_{SNRdb}dB", {"Perfect_H": perfect_dataset})
    savemat(f"data/Dataset_for_{SNRdb}db/Noisy_H_dataset_for_{SNRdb}dB", {"Noisy_H": noisy_dataset})
        
    
    

main()