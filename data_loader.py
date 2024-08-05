import torch 
from torch.utils.data import Dataset
import pandas as pd
from scipy.fft import fft, ifft
from scipy.signal import stft, istft
import random
import numpy as np 
import matplotlib.pyplot as plt 

random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_backgroundIdentification(f,t,original_spectrogram, original_signal, args):

    frequency_composition_abs = np.abs(original_spectrogram)
    measures = []
    for freq,freq_composition in zip(f,frequency_composition_abs):
        measures.append(np.mean(freq_composition)/np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    dummymatrix = np.zeros((len(f),len(t)))
    dummymatrix[selected_frequency,:] = 1  
    
    background_frequency = original_spectrogram * dummymatrix
    background_frequency = torch.tensor(background_frequency)
    _, xrec = istft(background_frequency,args.fs,nperseg=args.nperseg,noverlap=args.noverlap,boundary='zeros')
    xrec = xrec[:original_signal.shape[0]]
    xrec = xrec.reshape(original_signal.shape)
    return xrec, background_frequency

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, args):
        
        self.args = args
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].cpu().detach().numpy()

        #frequency
        freq_output = fft(data)

        #spectrogram
        f,t,spectrogram = stft(data, fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
        #obtain the rbp of spectrogram 
        xrec, background_spectrogram = sample_backgroundIdentification(f,t,spectrogram, self.data[idx], self.args) 
        noise_data = add_noise(self.data[idx])
        f_noise, t_noise, noise_spectrogram = stft(noise_data.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
         
        return self.data[idx], self.labels[idx], spectrogram, background_spectrogram, noise_data, noise_spectrogram, xrec, freq_output 

def load_data(data_name, args):
    
    print(f"Dataset is {data_name}")
    
    if data_name == "toydata_final":
        class0_data = torch.tensor(pd.read_csv('data/toydata_final/class1.csv').values)
        class1_data = torch.tensor(pd.read_csv('data/toydata_final/class2.csv').values)
        class2_data = torch.tensor(pd.read_csv('data/toydata_final/class3.csv').values)
        data = torch.cat([class0_data,class1_data, class2_data], dim=0)[:,:192]
        
        class0_label = torch.zeros(len(class0_data))
        class1_label = torch.ones(len(class1_data))
        class2_label = torch.ones(len(class0_data)) + 1

        labels = torch.cat((class0_label, class1_label, class2_label), dim=0)

    elif data_name == 'mixedshapes':
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/MixedShapesRegularTrain/MixedShapesRegularTrain_TRAIN.txt'  
        file_path2 = 'data/MixedShapesRegularTrain/MixedShapesRegularTrain_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)-1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)-1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0) 

    elif data_name=="yoga":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/Yoga/Yoga_TRAIN.txt'  
        file_path2 = 'data/Yoga/Yoga_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)-1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)-1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)  

    elif data_name=="forda":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/FordA/FordA_TRAIN.txt'  
        file_path2 = 'data/FordA/FordA_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)
        
        labels[labels == -1] = 0

    elif data_name=="fordb":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/FordB/FordB_TRAIN.txt'  
        file_path2 = 'data/FordB/FordB_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)
        
        labels[labels == -1] = 0

    elif data_name=="strawberry":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/Strawberry/Strawberry_TRAIN.txt'  
        file_path2 = 'data/Strawberry/Strawberry_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)-1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)-1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)  

    elif data_name=="cincecgtorso":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/CinCECGTorso/CinCECGTorso_TRAIN.txt'  
        file_path2 = 'data/CinCECGTorso/CinCECGTorso_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)-1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)-1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0) 

    elif data_name=="gunpointmalefemale":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/GunPointMaleVersusFemale/GunPointMaleVersusFemale_TRAIN.txt'  
        file_path2 = 'data/GunPointMaleVersusFemale/GunPointMaleVersusFemale_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)-1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)-1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)  

    elif data_name=="arrowhead":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/ArrowHead/ArrowHead_TRAIN.txt'  
        file_path2 = 'data/ArrowHead/ArrowHead_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)

    elif data_name=="twopatterns":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/TwoPatterns/TwoPatterns_TRAIN.txt'  
        file_path2 = 'data/TwoPatterns/TwoPatterns_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)-1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)-1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)

    else:
        print("Unknown Data name")
    
    #return num_freq and num_slices
    f,t,spectrogram = stft(data, fs=args.fs, nperseg=args.nperseg, noverlap=args.noverlap, boundary='zeros')
    
    num_freq = f.shape[0]
    num_slices = t.shape[0]
    
    print(f"Data Shape: {data.shape}, labels shape: {labels.shape}")
    
    ds = TimeSeriesDataset(data, labels, args)
    
    return ds, num_freq, num_slices 