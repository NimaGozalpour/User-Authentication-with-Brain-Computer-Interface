import os
import glob
import numpy as np
import _pickle as cPickle


from scipy import signal
from tqdm import tqdm
from scipy.fftpack import rfft, rfftfreq, irfft  #FFT for band frequency extraction

import argparse
from pathlib import Path

def read_timeseries(row):
    eeg_trial = row[0:32]
    return eeg_trial[:, 0:8064]

def load_data(data_dir):
    X = []
    labels = []
    metadata = []
    files = os.listdir(data_dir)
    for j, name in enumerate(files):
        filename = glob.glob(data_dir + '/' + name)
        if filename[0].endswith(".dat"):
            print(filename[0])
            all_trials_data = cPickle.load(
                open(filename[0], 'rb'), encoding='iso-8859-1')
            for i in range(len(all_trials_data['data'])):
                single_trial = read_timeseries(all_trials_data['data'][i])
                X.append(single_trial)
                labels.append(all_trials_data['labels'][i])
                metadata.append(np.array([int(filename[0].split('/')[-1].split('s')[1].split('.dat')[0]), i+1]))
                
    return np.array(X), np.array(labels), np.array(metadata)





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
        try:
            bandfft[0 , lowF : highF] = yf[lowF : highF]
        except:
             print(lowF,highF)
             
        #print(np.shape(bandfft))
        
        bandSignal[i, :] = irfft(bandfft, length)
    
    return bandSignal
    
    
def band_extractor_wrapper(data, band, samplingRate = 160, channels = 64):
    
    bandName = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    bands_frequency = {'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 12), 'beta': (13, 30), 'gamma': (30, 100)}

    band_to_num = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}
    bandNum = band_to_num[band]
    
    bandSignals = np.zeros((channels, len(bandName), data.shape[1]))
    for j in range(channels):
        bandSignals[j, :, :] = band_extractor(data[j, :], bands_frequency, samplingRate)
    
        
    return bandSignals[:, bandNum, :]


def get_samples(data, metadata, band, window, fs = 128, channels = 32, baseLine_seconds = 3, signal_length = 63):
    
    samples = []
    info = []
    shape = data.shape
        
    signal = data[:, baseLine_seconds * fs:]
    iden = metadata[0]
    task = metadata[1]
    
    for i in range(int((signal_length- baseLine_seconds)/(window))):
        feature = np.zeros((1 ,10 * channels))

        start = int(i * fs * window)
        end = int((i + 1) * fs * window)
        feature[0, : channels] = np.mean(signal[:, start : end], axis = 1)
        feature[0, channels: channels * 2] = np.std(signal[:, start : end], axis = 1)

        start = int(i * fs * window)
        end = int(i * fs * window + fs * window / 4)
        feature[0, channels * 2: channels * 3] = np.mean(signal[:, start : end], axis = 1)
        feature[0, channels * 3: channels * 4] = np.std(signal[:, start : end], axis = 1)

        start = int(i * fs * window + fs * window / 4)
        end = int(i * fs * window + 2 * fs * window / 4)
        feature[0, channels * 4: channels * 5] = np.mean(signal[:, start : end], axis = 1)
        feature[0, channels * 5: channels * 6] = np.std(signal[:, start : end], axis = 1)

        start = int(i * fs * window + 2 * fs * window / 4)
        end = int(i * fs * window + 3 * fs * window / 4)
        feature[0, channels * 6: channels * 7] = np.mean(signal[:, start : end], axis = 1)
        feature[0, channels * 7: channels * 8] = np.std(signal[:, start : end], axis = 1)

        start = int(i * fs * window + 3 * fs * window / 4)
        end = int((i + 1) * fs * window) 
        feature[0, channels * 8: channels * 9] = np.mean(signal[:, start : end], axis = 1)
        feature[0, channels * 9: channels * 10] = np.std(signal[:, start : end], axis = 1)

        samples.append(feature)
        info.append([iden, task, band, 'T4'])
        
                       
    return samples, info
    

    
def main(args):
    curr_dir = Path(__file__).parent.absolute()
    file_path = os.path.abspath(os.path.join(curr_dir, "..", "..", "rawData", "DEAP" , "data_preprocessed_python"))


    infos = []
    X = []

    files = os.listdir(file_path)
    for j, name in tqdm(enumerate(files), ascii = True, desc = 'Reading files'):
        filename = glob.glob(file_path + '/' + name)
        if filename[0].endswith(".dat"):
            
            all_trials_data = cPickle.load(
                open(filename[0], 'rb'), encoding='iso-8859-1')
            for i in range(len(all_trials_data['data'])):

                single_trial = read_timeseries(all_trials_data['data'][i])
                

                if args.band != 'allbands':
                    single_trial = band_extractor_wrapper(single_trial, args.band, samplingRate = 128, channels = 32)
                    
                #labels.append(all_trials_data['labels'][i])
                info = [int(filename[0].split('/')[-1].split('s')[1].split('.dat')[0]), i+1]
                
                samples, info = get_samples(single_trial, info, args.band, args.window)

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
    


