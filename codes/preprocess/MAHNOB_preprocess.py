import os

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

import mne
from mne.preprocessing import ICA
from scipy.fftpack import rfft, rfftfreq, irfft  #FFT for band frequency extraction


from sklearn.decomposition import PCA

import pyedflib  # https://pypi.org/project/pyEDFlib/

import argparse
from pathlib import Path
from tqdm import tqdm

def get_files(dir):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

def get_dirs(dir):
    return [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]


def extract_valence_arousal_score(xml_path):
    # parse the XML file
    tree = ET.parse(xml_path)
    # get the root element
    root = tree.getroot()
    # extract the value of the feltArsl attribute
    feltArsl = root.attrib['feltArsl']
    feltVlnc = root.attrib['feltVlnc']
    return {'valence': feltVlnc, 'arousal': feltArsl}


def downsample_time_series(time_series, original_sampling_rate=256, target_sampling_rate=128):
    decimation_factor = int(original_sampling_rate / target_sampling_rate)
    truncated_length = len(time_series) - \
        (len(time_series) % decimation_factor)
    truncated_time_series = time_series[:truncated_length]
    return np.mean(truncated_time_series.reshape(-1, decimation_factor), axis=1)


def read_blink_column(csv_file):
    df = pd.read_csv(csv_file)
    blink_data = np.array(df['blink'])
    return blink_data


def remove_blink_effects(eeg_signal, blink_flags, eeg_sampling_rate=128):
    # Convert the blink flags to the same sampling rate as the EEG signal
    blink_flags_resampled = signal.resample(blink_flags, len(eeg_signal))
    # Interpolate the blink flags
    interp_func = interp1d(np.arange(len(eeg_signal)) / eeg_sampling_rate,
                           blink_flags_resampled, kind='zero', bounds_error=False, fill_value=(0, 0))
    blink_interpolated = interp_func(
        np.arange(len(eeg_signal)) / eeg_sampling_rate)
    # Create a blink filter using a Gaussian window
    blink_filter_size = int(eeg_sampling_rate * 0.01)  # use a 100ms window
    blink_filter = np.exp(-0.5 * np.linspace(-2, 2, blink_filter_size) ** 2)
    # Convolve the blink filter with the interpolated blink flags
    blink_convolved = np.convolve(
        blink_interpolated, blink_filter, mode='same')
    # Normalize the blink filter
    blink_filter /= np.sum(blink_filter)
    # Convolve the blink filter with the EEG signal
    eeg_convolved = np.convolve(eeg_signal, blink_filter, mode='same')
    # Subtract the convolved blink signal from the EEG signal
    eeg_no_blink = eeg_signal - eeg_convolved * blink_convolved
    return eeg_no_blink


def bandpass_filter(signal, lowcut=0.4, highcut=45, fs=128, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def apply_average_reference(eeg_data):
    return eeg_data - np.mean(eeg_data, axis=0)


def clean_eeg_data_using_ica(eeg_data, ch_names=None, sfreq=128):
    ch_types = ['eeg'] * eeg_data.shape[0]
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(eeg_data, info)
    montage_1020 = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage_1020)
    ica = ICA(n_components=32, random_state=42)
    ica.fit(raw)
    ica.apply(raw)
    cleaned_eeg = raw.get_data()
    return cleaned_eeg



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

# Artifact Subspace Reconstruction
def apply_asr(eeg_data, ch_names, sfreq=128):
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    raw = mne.io.RawArray(eeg_data, info, verbose = False)
    raw = raw.filter(1, 50, verbose = False)
    raw = raw.notch_filter(60, method='spectrum_fit', verbose = False)

    # Compute channel-wise mean and standard deviation
    X = raw.get_data()
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)

    # Center the data
    X_centered = X - X_mean

    # Apply PCA
    pca = PCA(n_components=len(ch_names))
    pca.fit(X_centered.T)
    W = pca.components_

    # Compute projection vectors
    V = np.dot(W, np.diag(1 / np.sqrt(pca.explained_variance_)))
    proj = np.eye(len(ch_names)) - np.dot(V, V.T)

    # Apply projection matrix
    raw._data[:] = np.dot(proj, raw.get_data())

    return raw.get_data()


def preprocess_eeg_signals(filepath, method='ASR',
                           ch_names=['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']):
    f = pyedflib.EdfReader(filepath)

    sigbufs = np.zeros((32, f.getNSamples()[0]))
    for i in range(0, 32):
        sigbufs[i, :] = f.readSignal(i)

    sampling_rate = 256

    padding_before_start = 0
    padding_before_end = 30*sampling_rate

    padding_after_start = (sigbufs.shape[1] - 1) - 30*sampling_rate
    padding_after_end = (sigbufs.shape[1] - 1)

    stimuli_start = padding_before_end
    stimuli_end = padding_after_start

    #print(padding_before_start, padding_before_end, stimuli_start,
    #      stimuli_end, padding_after_start, padding_after_end)

    # (1) Downsampled to  128
    trial_signals_downsampled = []
    for i in range(32):
        downsampled = downsample_time_series(
            sigbufs[i][stimuli_start: stimuli_end], original_sampling_rate=256, target_sampling_rate=128)
        trial_signals_downsampled.append(downsampled)
    trial_signals_downsampled = np.array(trial_signals_downsampled)
    #print('(1) ------------------------------> Downsampled to  128!')

    # (2) Artifacts Removal
    trial_signals_artifacts_removed = None
    if method == 'ICA':
        trial_signals_artifacts_removed = clean_eeg_data_using_ica(
            eeg_data=trial_signals_downsampled, sfreq=128, ch_names=ch_names)
    elif method == 'ASR':
        trial_signals_artifacts_removed = apply_asr(
            eeg_data=trial_signals_downsampled, ch_names=ch_names)
    #print('(2) ------------------------------> Artifacts removed!')

    # (3) Bandpass filter 4 - 45 Hz
    trial_signals_bandpassed = []
    for i in range(32):
        trial_signals_bandpassed.append(bandpass_filter(
            signal=trial_signals_artifacts_removed[i], fs=128))
    trial_signals_bandpassed = np.array(trial_signals_bandpassed)
    #print('(3) ------------------------------> Bandpass filter 0.4 - 45 Hz applied!')

    # (4) Average to the common reference
    averaged_signal = apply_average_reference(trial_signals_bandpassed)
    #print('(4) ------------------------------> Averaged to the common reference!')

    return averaged_signal





def get_samples(data, metadata, band, window, fs = 128, channels = 32, baseLine_seconds = 0, signal_length = 60):
    
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
    file_path = os.path.abspath(os.path.join(curr_dir, "..", "..", "rawData", "MAHNOB" , "Data", 'data', 'Sessions'))
    dirs = get_dirs(file_path)

    infos = []
    X = []
    method = 'ASR'
    
    
    for dir in tqdm(dirs):
        trial_folder_path = os.path.join(file_path, dir)
        files = get_files(trial_folder_path)
        if len([file for file in files if file.endswith('.bdf')]) == 0:
            continue
        if len([file for file in files if file.endswith('.xml')]) == 0:
            continue

        bdf_file_name = [file for file in files if file.endswith('.bdf')][0]
        bdf_file_path = os.path.join(trial_folder_path, bdf_file_name)

        xml_file_name = [file for file in files if file.endswith('.xml')][0]
        xml_file_path = os.path.join(trial_folder_path, xml_file_name)

        #print("<><><><><><><><><><><><><><><><><><><><>\n",bdf_file_path, "<><><><><><><><><><><><><><><><><><><><>\n",)
        #scores = extract_valence_arousal_score(xml_file_path)
        #valence.append(scores['valence'])
        #arousal.append(scores['arousal'])
        #print(scores)

        single_trial = preprocess_eeg_signals(bdf_file_path, method=method)
        if single_trial.shape[1] < 7680:
            continue
        single_trial = single_trial[:7680]

        if args.band != 'allbands':
            single_trial = band_extractor_wrapper(single_trial, args.band, samplingRate = 128, channels = 32)
        
        path = os.path.join(dir, bdf_file_name)
        info = [int(path[path.find('Part_') + 5 : path.find('_S')]), int(path[path.find('Trial') + 5 : path.find('_emotion')])]
        
        
        samples, info = get_samples(single_trial, info, args.band, args.window)

        X += samples
        infos += info
        
    X = np.array(X)
    print(X.shape, len(infos))
    return X, infos
   


if __name__ == "__main__":

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
    

    
