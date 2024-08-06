import sys 
import torch 
import numpy as np
import pandas as pd
from scipy.signal import stft, istft
from scipy.fft import fft, ifft
from data_loader import *
from models.models import *
from utils.util import *   
import math 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import itertools

torch.set_num_threads(32)
random.seed(42)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Metric_Eval():
    def __init__(self, args, net):
        
        self.args = args
        self.model = net.to(device)
        a = torch.load(self.args.classification_model)
        self.model.load_state_dict(a['model_state_dict'])
        
        for param in self.model.parameters():
            param.requires_grad = False        
        
        self.model.eval()    
        
        # Initialize and seed the generator
        self.generator = torch.Generator()
        self.generator.manual_seed(911)        
        
        ds, height, width = load_data(self.args.dataset, self.args)
        groups = generate_region_groups(height, width, 1, 1)
        self.height = height
        self.width = width
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size], generator=self.generator)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
        
        #xai 
        self.label = self.args.label
        self.num_per = self.args.num_perturbations
        self.selected_regions = self.args.selected_regions
        self.groups = groups

    def faithfulness_compute(self):
        total_masked_regions = self.args.ranking

        for n in range(len(total_masked_regions)):
            faithfulness_value_j = []
            for j in range(0, self.args.topk+1, self.args.step_size):
                if self.args.step_size == 1:
                    masked_regions = total_masked_regions[n][j]
                else:
                    if j == 0:
                        j = 1
                    masked_regions = total_masked_regions[n][:j]
                initial_probability_scores = []
                masked_probability_scores = []
                for _, (data, labels, spectrogram, _, _) in enumerate(self.test_loader):
                    #select the data,spectrogram depending on labels
                    data = data[labels == self.label].unsqueeze(1).float()
                    spectrogram = spectrogram[labels == self.label]
                    labels = labels[labels == self.label]
                    
                    spectrogram = spectrogram.unsqueeze(1).to(device)
                    data = data.to(device)
                    
                    #initial probability of original signal without mask
                    #apply forward pass
                    with torch.no_grad(): 
                        x= self.model(data.float())
                        
                    x = torch.softmax(x, dim=1)
                    initial_probability_scores.append(x[:,self.label])
                    #probability after masking top-k features in spectrogram of original signal 
                    #flatten spectrogram
                    spectrogram = spectrogram.view(spectrogram.shape[0], 1, -1)
                    
                    #apply masking
                    for i in masked_regions:
                        spectrogram[:,:,i] = 0

                    
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.height,self.width)
                    
                    #apply inverse stft 
                    _,inverse_stft = istft(spectrogram.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    masked_signal = torch.tensor(inverse_stft).to(device)

                    #apply forward pass
                    with torch.no_grad(): 
                        x_masked = self.model(masked_signal.float())
                    
                    x_masked = torch.softmax(x_masked, dim=1)
                    masked_probability_scores.append(x_masked[:,self.label])
                
                #calculate difference between initial and masked probability scores
                result = [a - b for a, b in zip(initial_probability_scores, masked_probability_scores)]
                result_tensor = torch.cat(result, dim=0)

                #determine the faithfulness value
                faithfulness_value = torch.mean(result_tensor)
                faithfulness_value_j.append(faithfulness_value.item())
            print(f"Total Faithfulness Value: {faithfulness_value_j}")    


             
            
            
            
            
            
            
            