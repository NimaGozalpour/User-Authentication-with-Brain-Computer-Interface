import pyedflib # https://pypi.org/project/pyEDFlib/
# Documentation: https://pyedflib.readthedocs.io/en/latest/

import numpy as np
import os
from scipy import signal
from tqdm import tqdm
import pickle
from scipy.fftpack import rfft, rfftfreq, irfft  #FFT for band frequency extraction
import argparse
from pathlib import Path








def low_high_pass_notch_filter(one_channel_signal, fs = 160 ,f_low = 0.5, f_high = 60):
    
    # Define the Butterworth filter low pass
    fc = f_high  # Hz
    b, a = signal.butter(4, fc/(fs/2), 'low')
    # Apply the low pass filter to the signal
    one_channel_signal = signal.filtfilt(b, a, one_channel_signal)
    
    
    fc = f_low  # Hz
    b, a = signal.butter(4, fc/(fs/2), 'high')
    # Apply the high pass filter to the signal
    one_channel_signal = signal.filtfilt(b, a, one_channel_signal)
    
    
    # Define the notch filter
    f0 = 50  # Hz
    Q = 10
    b, a = signal.iirnotch(f0, Q, fs)
    
    # Apply the notch filter to the signal
    one_channel_signal = signal.filtfilt(b, a, one_channel_signal)
    
    return one_channel_signal


#Frequency Band Extractor
def band_extractor(signal, bands_frequency, samplingRate = 128):
    length = len(signal)
    
    duration = length/(samplingRate)*1.0
    
    yf = rfft(signal)
    bandSignal = np.zeros((len(bands_frequency), length))
    
    
    for i, name in enumerate(bands_frequency):
        bandfft = np.zeros((1, length))
        #print(bands_frequency[name], bands_frequency[name][0])
        lowF = bands_frequency[name][0] * samplingRate
        highF = bands_frequency[name][1] * samplingRate
        if highF > length:
            highF = length
        bandfft[0 , lowF : highF] = yf[lowF : highF]
        
        bandSignal[i, :] = irfft(bandfft, length)
    
    return bandSignal
    
    
def band_extractor_wrapper(data, band, samplingRate = 160, channels = 64):
    
    bandName = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    bands_frequency = {'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 12), 'beta': (13, 30), 'gamma': (30, 100)}
    
    band_to_num = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}
    bandNum = band_to_num[band]


    bandSignals = np.zeros((channels, len(bandName), data[2].shape[1]))
    for j in range(channels):
        bandSignals[j, :, :] = band_extractor(data[2] [j, :], bands_frequency, samplingRate)
    
    

    return [data[0], data[1], bandSignals[:, bandNum, :], data[3]]



def read_raw_data_EDF(f, file_path):
    Signals = []
    
    
    #try:
    path = os.path.join(file_path, f[:4], f[5:-1])
    #print(path)
    h = pyedflib.EdfReader(path)
    n = h.signals_in_file
    sigbufs = np.zeros((n, h.getNSamples()[0]))

    n = h.signals_in_file
    for j in np.arange(n):
        sigbufs[j, :] = h.readSignal(j)
        sigbufs[j, :] = low_high_pass_notch_filter(sigbufs[j, :])

    Signals.append(int(f[1:4]))
    Signals.append(int(f[-7:-5]))
    #print(b)
    
    # Apply the Common average reference filter to the signal
    sigbufs = sigbufs - np.mean(sigbufs, axis=0)
    
    Signals.append(sigbufs)
    Signals.append(h.readAnnotations())

    
        
    return Signals



def get_samples(data, band, window, fs = 160):
    
    

    signal = data[2]
    annotation = data[3]
    iden = data[0]
    task = data[1]
    
    samples = []
    infos = []
    if task < 3:
        for i in range(int(60/(window))):
            state = 'T0'
            feature = np.zeros((1,10*64))
            
            start = int(i * 160 * window)
            end = int((i + 1) * 160 * window)
            feature[0, : 64] = np.mean(signal[:, start : end], axis = 1)
            feature[0, 64: 128] = np.std(signal[:, start : end], axis = 1)
            
            start = int(i * 160 * window)
            end = int(i * 160 * window + 160 * window / 4)
            feature[0, 64 * 2: 64 * 3] = np.mean(signal[:, start : end], axis = 1)
            feature[0, 64 * 3: 64 * 4] = np.std(signal[:, start : end], axis = 1)
            
            start = int(i * 160 * window + 160 * window / 4)
            end = int(i * 160 * window + 2 * 160 * window / 4)
            feature[0, 64 * 4: 64 * 5] = np.mean(signal[:, start : end], axis = 1)
            feature[0, 64 * 5: 64 * 6] = np.std(signal[:, start : end], axis = 1)
            
            start = int(i * 160 * window + 2 * 160 * window / 4)
            end = int(i * 160 * window + 3 * 160 * window / 4)
            feature[0, 64 * 6: 64 * 7] = np.mean(signal[:, start : end], axis = 1)
            feature[0, 64 * 7: 64 * 8] = np.std(signal[:, start : end], axis = 1)
            
            start = int(i * 160 * window + 3 * 160 * window / 4)
            end = int((i + 1) * 160 * window )
            feature[0, 64 * 8: 64 * 9] = np.mean(signal[:, start : end], axis = 1)
            feature[0, 64 * 9: 64 * 10] = np.std(signal[:, start : end], axis = 1)

            samples.append(feature)
            infos.append([iden, task, band, state])
            
    else:
        #print(annotation[0], len(annotation[0]))
        if len(annotation[0]) == 30:
            for count, value in enumerate(annotation[0]):
                state = annotation[2][count]

                k = 0
                while k < 4:

                    feature = np.zeros((1,10*64))

                    i = value + k
                    k += window
                    start = int(i * 160)  
                    end = int((i + window) * 160) 
                    feature[0, : 64] = np.mean(signal[:, start : end], axis = 1)
                    feature[0, 64: 128] = np.std(signal[:, start : end], axis = 1)

                    start = int(i * 160)  
                    end = int((i + window / 4) * 160) 
                    feature[0, 64 * 2: 64 * 3] = np.mean(signal[:, start : end], axis = 1)
                    feature[0, 64 * 3: 64 * 4] = np.std(signal[:, start : end], axis = 1)

                    start = int((i + window / 4) * 160) 
                    end = int((i + 2 * window / 4) * 160) 
                    feature[0, 64 * 4: 64 * 5] = np.mean(signal[:, start : end], axis = 1)
                    feature[0, 64 * 5: 64 * 6] = np.std(signal[:, start : end], axis = 1)

                    start = int((i + 2 * window / 4) * 160) 
                    end = int((i + 3 * window / 4) * 160) 
                    feature[0, 64 * 6: 64 * 7] = np.mean(signal[:, start : end], axis = 1)
                    feature[0, 64 * 7: 64 * 8] = np.std(signal[:, start : end], axis = 1)

                    start = int((i + 3 * window / 4) * 160) 
                    end = int((i + window) * 160) 
                    feature[0, 64 * 8: 64 * 9] = np.mean(signal[:, start : end], axis = 1)
                    feature[0, 64 * 9: 64 * 10] = np.std(signal[:, start : end], axis = 1)

                    samples.append(feature)
                    infos.append([iden, task, band, state])
            

        #different setup(duration of execuation of task different therefore discard these trials)            
        else:
            return False, False, False
            
    
    
    return samples, infos, True
    
    

    
def main(args):
    curr_dir = Path(__file__).parent.absolute()
    file_path = os.path.abspath(os.path.join(curr_dir, "..", "..", "rawData", "EMMI" , "files", "eegmmidb", "1.0.0"))

    
    f = open(os.path.join(file_path,"RECORDS"), "r")

    infos = []
    X = []

    for i in tqdm(f):

        single_trial = read_raw_data_EDF(i, file_path)
    
        if args.band != 'allbands':
            single_trial = band_extractor_wrapper(single_trial, args.band)
    
        samples, info, state = get_samples(single_trial, args.band, args.window)    

        if state:
            X += samples
            infos += info
            
    X = np.array(X)
    print(X.shape, len(infos))
    return X, infos


if __name__ == '__main__':

    # Configure CLI parser early. This way we don't need to load TF if there's a missing arg.
    parser = argparse.ArgumentParser(description='different scenario',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Define CLI options.
    parser.add_argument('--band', default= 'allbands', help='Band Name: allbands ,delta, theta, alpha, beta, gamma')
    
    parser.add_argument('--window', default = 4, type = float, help='Window size of sampling')
    
    parser.add_argument('--UnicornPlacement', default = 'False', help='True or False')
    
    parser.add_argument('--Action', default = 'True', help='True or Flase')
    
    parser.add_argument('--TaskDependent', default = 'True', help='True or Flase, True means random, otherwise specifiy TCV and TT')
    
    parser.add_argument('--TaskNum', default = 0, type = int, help='0 means all task, for individual task enter number of that task example: 1, 2, 3,...,14')
    
    parser.add_argument('--TCV', type=int, nargs='+', default=[11, 12], help='Tasks for cross validation')
    parser.add_argument('--TT', type=int, nargs='+', default=[13, 14], help='Tasks for test')
    parser.add_argument('--RN', type=int, default = 1, help='Run Number')
    parser.add_argument('--maxEpoch', type=int, default = 220, help='Max Epoch must be integer default is 220')
    parser.add_argument('--dataset', default = 'EMMI', help='EMMI, DEAP, MAHNOB')
    parser.add_argument('--batchSize', default = 256, type = int ,help='EMMI, DEAP,(256 good) MAHNOB(64 good)')
    parser.add_argument('--verbose', default = 0 , type = int , help='0, 1')


    args = parser.parse_args()



    main(args)
    










