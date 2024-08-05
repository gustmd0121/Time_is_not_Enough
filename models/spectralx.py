import torch 
import pandas as pd 
import random
from collections import Counter
import data_loader 
from scipy.fft import fft, ifft
from models.models import *
from utils.util import *


torch.set_num_threads(32)
random.seed(42)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XAITrainer():
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

        ds = data_loader(self.args.dataset, self.args)
        ds, width, height = data_loader(self.args.dataset, self.args)
        groups = generate_region_groups(1, 1, height, width)
        self.args.groups = groups
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
        self.noise = self.args.noisy
    
    def insertion(self, selected_positions, positions_consider):
        inverse_data = []
        position_scores = []
        indices_probability_scores = []
        class_probability_scores = list(0. for i in range(self.num_per))
        total_count = 0
        for n, (data, labels, spectrogram, rbp_spec, noise_data, noise_spectrogram, rbp_signal, freq_output) in enumerate(self.test_loader):
            #check if prediction and labels match
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float() 
            data_check, labels = data_check.to(device), labels.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)
            
            if self.args.just_correct:
                #select the data,spectrogram depending on labels
                noise_spectrogram = noise_spectrogram[(labels == self.label) & (predicted == self.label)]
                noise_data = noise_data[(labels == self.label) & (predicted == self.label)]
                data = data[(labels == self.label) & (predicted == self.label)].to(device)
                spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
                rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
                rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
                freq_output = freq_output[(labels == self.label) & (predicted == self.label)]  
                labels = labels[(labels == self.label) & (predicted == self.label)]
            else:
                noise_spectrogram = noise_spectrogram[(labels == self.label)]
                noise_data = noise_data[(labels == self.label)]
                data = data[(labels == self.label)].to(device)
                spectrogram = spectrogram[(labels == self.label)]
                rbp_spec = rbp_spec[(labels == self.label)]
                rbp_signal = rbp_signal[(labels == self.label)]
                freq_output = freq_output[(labels == self.label)]
                labels = labels[(labels == self.label)] 
            
            if self.noise:
                data = noise_data.to(device)
                spectrogram = noise_spectrogram
            
            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64).to(device)

            if self.args.domain == 'time':
                masked_tensor = rbp_signal.to(device)
            elif self.args.domain == 'freq':
                masked_tensor = torch.zeros_like(freq_output)
                masked_tensor[:, 0] = freq_output[:, 0]
            elif self.args.masking_method == 'zero':
                masked_tensor = torch.zeros_like(spectrogram)
            elif self.args.masking_method == 'class_rbp':
                background_data, masked_tensor = backgroundIdentification(data.cpu().detach().numpy(), self.args)
                masked_tensor = masked_tensor.to(torch.complex64).to(device)
            elif self.args.masking_method == 'sample_rbp':
                masked_tensor = rbp_spec.to(torch.complex64).to(device)
            
            # #initial blank spectrogram
            # initial_masked_tensor = masked_tensor.clone()
            # _,initial_inverse_stft = istft(initial_masked_tensor.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
            # initial_x = torch.tensor(initial_inverse_stft).unsqueeze(1).to(device)

            # #apply forward pass
            # with torch.no_grad(): 
            #     initial_x, _,_,_,_ = self.model(initial_x.float())
            
            # initial_x = torch.softmax(initial_x, dim=1)
    
            if not selected_positions: 
                #generate random indices
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions) 
                positions_consider = self.groups 
                combinations = positions_consider 
            
                for i in range(self.num_per):
                    new_tensor = masked_tensor.clone()
                    
                    #flatten data, new_tensor and spectrogram
                    data = data.view(data.shape[0], 1, -1).to(torch.float64)
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    spectrogram = spectrogram.view(spectrogram.shape[0], 1, -1)
                    freq_output = freq_output.view(freq_output.shape[0], 1, -1)
                    
                    #all combinations of positions to consider 
                    for j in [combinations[a] for a in indices_list[i]]:
                        #insert feature into new_tensor
                        if self.args.domain == 'time':
                            new_tensor[:, :, j] = data[:, :, j]
                        elif self.args.domain == 'freq':
                            mid_index = freq_output.shape[-1] // 2
                            symmetrical_index = mid_index + (mid_index - j[0])
                            new_tensor[:, :, j] = freq_output[:,:, j]
                            new_tensor[:, :, symmetrical_index] = freq_output[:, :, symmetrical_index]
                        elif self.args.domain == 'timefreq':
                            new_tensor[:, :, j] = spectrogram[:, :, j]
                    if self.args.domain == 'timefreq':
                        new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.args.num_freq,self.args.num_slices)
                    
                    x = new_tensor.to(device)

                    #apply inverse stft 
                    if self.args.domain == 'timefreq':
                        _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x = torch.tensor(inverse_stft).to(device)
                    elif self.args.domain == 'freq':
                        inverse_fft = ifft(x.cpu().detach().numpy())
                        inverse_fft = np.real(inverse_fft)
                        x = torch.tensor(inverse_fft).to(device)
                    
                    #save data 
                    # inverse_data.append(x)
                    
                    #apply forward pass
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    # x = x-initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])

                #save the inverse data
                # inverse_data = torch.stack(inverse_data)
                # torch.save(inverse_data, self.args.savedir + '/' + self.args.dataset + '_inverse_data_insertion.pt')
                # torch.save(torch.tensor(indices_list), self.args.savedir + '/' + self.args.dataset + '_indices_list_insertion.pt')
            else:
                #insert selected positions
                new_tensor = masked_tensor.clone()
                for position in selected_positions:
                    data = data.reshape(data.shape[0],1,-1).to(torch.float64)
                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,-1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,-1)
                    freq_output = freq_output.view(freq_output.shape[0], 1, -1)
                    if self.args.domain == 'time':
                        new_tensor[:,:,position] = data[:,:,position]
                    elif self.args.domain == 'freq':
                        mid_index = freq_output.shape[-1] // 2
                        symmetrical_index = mid_index + (mid_index - position[0])
                        new_tensor[:, :, position] = freq_output[:,:, position]
                        new_tensor[:, :, symmetrical_index] = freq_output[:, :, symmetrical_index]
                    else:
                        new_tensor[:,:,position] = spectrogram[:,:,position]
                        new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.args.num_freq,self.args.num_slices)

                #initial blank spectrogram after insertion
                # initial_masked_tensor = new_tensor.clone()
                # _,initial_inverse_stft = istft(initial_masked_tensor.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                # initial_x = torch.tensor(initial_inverse_stft).to(device)

                #apply forward pass
                # with torch.no_grad(): 
                #     initial_x = self.model(initial_x.float())
                
                # initial_x = torch.softmax(initial_x, dim=1)


                if n == 0:
                    #remove selected positions from positions to consider
                    position_list = positions_consider
                    designated_positions = selected_positions
                    
                    for sublist in designated_positions:
                        if sublist in position_list: 
                            position_list.remove(sublist)
                    
                    combinations = position_list #after removing selected regions
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)
                
                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,-1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,-1)
                    freq_output = freq_output.view(freq_output.shape[0], 1, -1)
                    for j in [combinations[a] for a in indices_list[i]]:
                        if self.args.domain == 'time':
                            altered_tensor[:,:,j] = data[:,:,j]
                        elif self.args.domain == 'timefreq':
                            altered_tensor[:,:,j] = spectrogram[:,:,j]
                        else:
                            mid_index = freq_output.shape[-1] // 2
                            symmetrical_index = mid_index + (mid_index - j[0])
                            altered_tensor[:,:,j] = freq_output[:,:,j]
                            altered_tensor[:, :, symmetrical_index] = freq_output[:, :, symmetrical_index]
                    if self.args.domain == 'timefreq':
                        altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.args.num_freq,self.args.num_slices)
                        
                    x = altered_tensor.to(device) 
                    #apply inverse stft 
                    if self.args.domain == 'timefreq':
                        _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x = torch.tensor(inverse_stft).to(device)
                    elif self.args.domain == 'freq':
                        inverse_fft = ifft(x.cpu().detach().numpy())
                        inverse_fft = np.real(inverse_fft)
                        x = torch.tensor(inverse_fft).to(device)
                    
                    #apply forward pass
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    # x = x - initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])
            
            total_count += data.shape[0]
        
        class_probability_scores = [x / total_count for x in class_probability_scores]
        count = Counter([index for sublist in indices_list for index in sublist])
        summed_scores_for_indices = sum_scores_for_each_index(indices_list, class_probability_scores)
        for i in range(len(summed_scores_for_indices)):
            indices_probability_scores.append(summed_scores_for_indices[i] / count[i])
        max_position = torch.argmax(torch.tensor(indices_probability_scores)) 
        max_position = combinations[max_position]

        selected_positions.append(max_position)
        return selected_positions, positions_consider
                    
    def deletion(self, selected_positions, positions_consider):
        
        inverse_data = []
        position_scores = []
        indices_probability_scores = []
        class_probability_scores = list(0. for i in range(self.num_per))
        total_count = 0

        for n, (data, labels, spectrogram, rbp_spec, noise_data, noise_spectrogram, rbp_signal, freq_output) in enumerate(self.test_loader):
            #check if prediction and labels match
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float() 
            data_check, labels = data_check.to(device), labels.to(device)
            output_check= self.model(data_check)
            _, predicted = torch.max(output_check, 1)
            
            if self.args.just_correct:
                #select the data,spectrogram depending on labels
                noise_spectrogram = noise_spectrogram[(labels == self.label) & (predicted == self.label)]
                noise_data = noise_data[(labels == self.label) & (predicted == self.label)]
                data = data[(labels == self.label) & (predicted == self.label)].to(device)
                spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
                rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
                rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)] 
                freq_output = freq_output[(labels == self.label) & (predicted == self.label)]  
                labels = labels[(labels == self.label) & (predicted == self.label)]
            else:
                noise_spectrogram = noise_spectrogram[(labels == self.label)]
                noise_data = noise_data[(labels == self.label)]
                data = data[(labels == self.label)].to(device)
                spectrogram = spectrogram[(labels == self.label)]
                rbp_spec = rbp_spec[(labels == self.label)]
                rbp_signal = rbp_signal[(labels == self.label)]
                freq_output = freq_output[(labels == self.label)]
                labels = labels[(labels == self.label)] 
            
            if self.noise:
                data = noise_data.to(device)
                spectrogram = noise_spectrogram
            
            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64).to(device)
            if self.args.domain == 'time':
                masked_tensor = rbp_signal.to(device)
            elif self.args.domain == 'freq':
                masked_tensor = torch.zeros_like(freq_output)
                masked_tensor[:, 0] = freq_output[:, 0]
            elif self.args.masking_method == 'zero':
                masked_tensor = torch.zeros_like(spectrogram)
            elif self.args.masking_method == 'class_rbp':
                background_data, masked_tensor = backgroundIdentification(data.cpu().detach().numpy(), self.args)
                masked_tensor = masked_tensor.to(torch.complex64).to(device) 
            elif self.args.masking_method == 'sample_rbp':
                masked_tensor = rbp_spec.to(torch.complex64).to(device)

            # plt.figure(figsize=(6, 6))
            # plt.imshow(np.abs(masked_tensor[0].cpu().detach().numpy()[:,:9]), aspect='auto', origin='lower')
            # plt.gca().set_xticks([])
            # plt.gca().set_yticks([])
            # plt.savefig("/home/hschung/xai/new_xai/plots/figure_plots/rbp_domain.png")

            if not selected_positions: 
                #generate random indices
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions) 
                positions_consider = self.groups 
                combinations = positions_consider 
                for i in range(self.num_per):
                    if self.args.domain == 'time':
                        new_tensor = data.clone().to(torch.float64)
                    elif self.args.domain == 'timefreq':
                        new_tensor = spectrogram.clone()
                    else:
                        new_tensor = freq_output.clone()
                    
                    #flatten new_tensor and spectrogram 
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    masked_tensor= masked_tensor.view(masked_tensor.shape[0], 1, -1)
                    
                    #all combinations of positions to consider 
                    for j in [combinations[a] for a in indices_list[i]]:
                        #insert feature into new_tensor 
                        new_tensor[:, :, j] = masked_tensor[:, :, j]
                        if self.args.domain == 'freq':
                            mid_index = freq_output.shape[-1] // 2
                            symmetrical_index = mid_index + (mid_index - j[0])
                            new_tensor[:, :, symmetrical_index] = masked_tensor[:, :, symmetrical_index]

                    if self.args.domain == 'timefreq':
                        new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                    
                    x = new_tensor.to(device)

                    #apply inverse stft 
                    if self.args.domain == 'timefreq':
                        _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x = torch.tensor(inverse_stft).to(device)
                    elif self.args.domain == 'freq':
                        inverse_fft = ifft(x.cpu().detach().numpy())
                        inverse_fft = np.real(inverse_fft)
                        x = torch.tensor(inverse_fft).to(device)
                    
                    #save data 
                    # inverse_data.append(x)
                    
                    #apply forward pass
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])

                #save the inverse data
                # inverse_data = torch.stack(inverse_data)
                # torch.save(inverse_data, self.args.savedir + '/' + self.args.dataset + f'_inverse_data_deletion_{self.label}.pt')
                # torch.save(torch.tensor(indices_list), self.args.savedir + '/' + self.args.dataset + f'_indices_list_deletion_{self.label}.pt')

            else:
                #insert selected positions
                if self.args.domain == 'time':
                    new_tensor = data.clone().to(torch.float64)
                elif self.args.domain == 'timefreq':
                    new_tensor = spectrogram.clone()
                else:
                    new_tensor = freq_output.clone()

                for position in selected_positions:
                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,-1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,-1)
                    new_tensor[:,:,position] = masked_tensor[:,:,position]
                    if self.args.domain == 'freq':
                        mid_index = freq_output.shape[-1] // 2
                        symmetrical_index = mid_index + (mid_index - position[0])
                        new_tensor[:, :, symmetrical_index] = masked_tensor[:, :, symmetrical_index]               
                    elif self.args.domain == 'timefreq':
                        new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)

                if n == 0:
                    #remove selected positions from positions to consider
                    position_list = positions_consider
                    designated_positions = selected_positions
                    
                    for sublist in designated_positions:
                        if sublist in position_list: 
                            position_list.remove(sublist)
                    
                    combinations = position_list #after removing selected regions
                    
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)

                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,-1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,-1)
                    for j in [combinations[a] for a in indices_list[i]]:
                        altered_tensor[:,:,j] = masked_tensor[:,:,j]
                        if self.args.domain == 'freq':
                            mid_index = freq_output.shape[-1] // 2
                            symmetrical_index = mid_index + (mid_index - j[0])
                            altered_tensor[:, :, symmetrical_index] = masked_tensor[:, :, symmetrical_index]
                    if self.args.domain == 'timefreq':
                        altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)

                    x = altered_tensor.to(device) 

                    if self.args.domain == 'timefreq':
                        #apply inverse stft 
                        _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x = torch.tensor(inverse_stft).to(device)
                    elif self.args.domain == 'freq':
                        inverse_fft = ifft(x.cpu().detach().numpy())
                        inverse_fft = np.real(inverse_fft)
                        x = torch.tensor(inverse_fft).to(device)
                    
                    #apply forward pass
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])

            total_count += data.shape[0]
        
        class_probability_scores = [x / total_count for x in class_probability_scores]
        count = Counter([index for sublist in indices_list for index in sublist])
        summed_scores_for_indices = sum_scores_for_each_index(indices_list, class_probability_scores)
        for i in range(len(summed_scores_for_indices)):
            indices_probability_scores.append(summed_scores_for_indices[i] / count[i])
        min_position = torch.argmin(torch.tensor(indices_probability_scores)) 
        min_position = combinations[min_position]

        selected_positions.append(min_position)
        return selected_positions, positions_consider         
                    
    def combined(self, selected_positions, positions_consider):
        
        position_scores = []
        indices_probability_scores = []
        indices_probability_scores_del = []
        class_probability_scores = list(0. for i in range(self.num_per))
        class_probability_scores_del = list(0. for i in range(self.num_per))
        total_count = 0

        for n, (data, labels, spectrogram, rbp_spec, noise_data, noise_spectrogram, rbp_signal, freq_output) in enumerate(self.test_loader):
            
            #check if prediction and labels match
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float() 
            data_check, labels = data_check.to(device), labels.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)
            
            labels = labels.to('cpu')
            
            #select the data,spectrogram depending on labels
            if self.args.just_correct:
                #select the data,spectrogram depending on labels
                noise_spectrogram = noise_spectrogram[(labels == self.label) & (predicted == self.label)]
                noise_data = noise_data[(labels == self.label) & (predicted == self.label)]
                data = data[(labels == self.label) & (predicted == self.label)].to(device)
                spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
                rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
                rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
                freq_output = freq_output[(labels == self.label) & (predicted == self.label)]   
                labels = labels[(labels == self.label) & (predicted == self.label)]
            else:
                noise_spectrogram = noise_spectrogram[(labels == self.label)].to(device)
                data = data[(labels == self.label)].to(device)
                spectrogram = spectrogram[(labels == self.label)].to(device)
                rbp_spec = rbp_spec[(labels == self.label)].to(device)
                rbp_signal = rbp_signal[(labels == self.label)].to(device)
                freq_output = freq_output[(labels == self.label)].to(device)
                labels = labels[(labels == self.label)].to(device) 

            if self.noise:
                data = noise_data.to(device)
                spectrogram = noise_spectrogram
            
            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64).to(device)
            spectrogram_del = spectrogram.clone()

            if self.args.domain == 'time':
                masked_tensor = rbp_signal.to(device)
                masked_tensor_del = masked_tensor.clone()
            elif self.args.domain == 'freq':
                masked_tensor = torch.zeros_like(freq_output)
                masked_tensor[:, 0] = freq_output[:, 0]
                masked_tensor_del = masked_tensor.clone()
            elif self.args.masking_method == 'zero':
                masked_tensor = torch.zeros_like(spectrogram)
                masked_tensor_del = torch.zeros_like(spectrogram_del)
            elif self.args.masking_method == 'class_rbp':
                background_data, masked_tensor = backgroundIdentification(data.cpu().detach().numpy(), self.args)
                masked_tensor = masked_tensor.to(torch.complex64).to(device)
                masked_tensor_del = masked_tensor.clone() 
            elif self.args.masking_method == 'sample_rbp':
                masked_tensor = rbp_spec.to(torch.complex64).to(device)
                masked_tensor_del = masked_tensor.clone()
           
            #initial blank spectrogram
            # initial_masked_tensor = masked_tensor.clone()
            # _,initial_inverse_stft = istft(initial_masked_tensor.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
            # initial_x = torch.tensor(initial_inverse_stft).unsqueeze(1).to(device)

            # #apply forward pass
            # with torch.no_grad(): 
            #     initial_x, _,_,_,_ = self.model(initial_x.float())
            
            # initial_x = torch.softmax(initial_x, dim=1)

            if not selected_positions: 
                #generate random indices
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions) 
                positions_consider = self.groups 
                combinations = positions_consider 
                data = data.view(data.shape[0], 1, -1).to(torch.float64)
                for i in range(self.num_per):
                    new_tensor = masked_tensor.clone()
                    if self.args.domain == 'time':
                        new_tensor_del = data.clone()
                    elif self.args.domain == 'timefreq':
                        new_tensor_del = spectrogram_del.clone()
                    else:
                        new_tensor_del = freq_output.clone()
                    #flatten new_tensor and spectrogram 
                    freq_output = freq_output.view(freq_output.shape[0], 1, -1)
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    spectrogram = spectrogram.view(spectrogram.shape[0], 1, -1)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0],1,-1)
                    masked_tensor_del = masked_tensor_del.reshape(masked_tensor_del.shape[0],1,-1)
                    
                    #all combinations of positions to consider 
                    for j in [combinations[a] for a in indices_list[i]]:
                        #insert feature into new_tensor 
                        if self.args.domain == 'time':
                            new_tensor[:, :, j] = data[:, :, j]
                            new_tensor_del[:,:,j] = masked_tensor_del[:,:,j]
                        elif self.args.domain == 'timefreq':
                            new_tensor[:, :, j] = spectrogram[:, :, j]
                            new_tensor_del[:,:,j] = masked_tensor_del[:,:,j]
                        else:
                            mid_index = freq_output.shape[-1] // 2
                            new_tensor[:, :, j] = freq_output[:, :, j]
                            new_tensor_del[:,:,j] = masked_tensor_del[:,:,j]
                            symmetrical_index = mid_index + (mid_index - j[0]) 
                            new_tensor[:, :, symmetrical_index] = freq_output[:, :, symmetrical_index] 
                            new_tensor_del[:, :, symmetrical_index] = masked_tensor_del[:, :, symmetrical_index]   

                    if self.args.domain == 'timefreq':
                        new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.args.num_freq,self.args.num_slices)
                        new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0],1,self.args.num_freq,self.args.num_slices)
                        masked_tensor_del = masked_tensor_del.reshape(masked_tensor_del.shape[0],1,self.args.num_freq,self.args.num_slices)
                    
                    x = new_tensor.to(device)
                    x_del = new_tensor_del.to(device)

                    #apply inverse stft
                    if self.args.domain == 'timefreq': 
                        _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x = torch.tensor(inverse_stft).to(device)        
                        _,inverse_stft_del = istft(x_del.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x_del = torch.tensor(inverse_stft_del).to(device)                
                    elif self.args.domain == 'freq':
                        inverse_fft = ifft(x.cpu().detach().numpy())
                        inverse_fft = np.real(inverse_fft)
                        x = torch.tensor(inverse_fft).to(device)

                        inverse_fft_del = ifft(x_del.cpu().detach().numpy())
                        inverse_fft_del = np.real(inverse_fft_del)
                        x_del= torch.tensor(inverse_fft_del).to(device)

                    #apply forward pass
                    with torch.no_grad(): 
                        x= self.model(x.float())
                        x_del= self.model(x_del.float())
                    
                    x = torch.softmax(x, dim=1)
                    # x = x-initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])
                    x_del = torch.softmax(x_del, dim=1)
                    class_probability_scores_del[i] = torch.sum(x_del[:,self.label])
            
            else:
                #insert selected positions
                new_tensor = masked_tensor.clone()
                if self.args.domain == 'time':
                    new_tensor_del = data.clone().to(torch.float64)
                elif self.args.domain == 'timefreq':
                    new_tensor_del = spectrogram.clone()
                else:
                    new_tensor_del = freq_output.clone()

                for position in selected_positions:
                    freq_output = freq_output.view(freq_output.shape[0], 1, -1)
                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,-1)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0],1,-1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,-1)
                    data = data.reshape(data.shape[0],1,-1).to(torch.float64)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,-1)
                    if self.args.domain == 'time':
                        new_tensor[:,:,position] = data[:,:,position]
                        new_tensor_del[:,:,position] = masked_tensor[:,:,position]
                    elif self.args.domain == 'freq':
                        mid_index = freq_output.shape[-1] // 2
                        new_tensor[:,:,position] = freq_output[:,:,position]
                        new_tensor_del[:,:,position] = masked_tensor[:,:,position]
                        symmetrical_index = mid_index + (mid_index - position[0]) 
                        new_tensor[:, :, symmetrical_index] = freq_output[:, :, symmetrical_index]
                        new_tensor_del[:, :, symmetrical_index] = masked_tensor[:, :, symmetrical_index]                      
                    elif self.args.domain == 'timefreq':
                        new_tensor[:,:,position] = spectrogram[:,:,position]
                        new_tensor_del[:,:,position] = masked_tensor[:,:,position]
                        new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.args.num_freq,self.args.num_slices)
                        new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0],1,self.args.num_freq, self.args.num_slices)
                        masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,self.args.num_freq, self.args.num_slices)                   

                # #initial spectrogram after insertion / deletion
                # initial_masked_tensor = new_tensor.clone()
                # _,initial_inverse_stft = istft(initial_masked_tensor.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                # initial_x = torch.tensor(initial_inverse_stft).to(device)
                # #apply forward pass
                # with torch.no_grad(): 
                #     initial_x, _,_,_,_ = self.model(initial_x.float())
                
                # initial_x = torch.softmax(initial_x, dim=1)

                if n == 0 :
                    #remove selected positions from positions to consider
                    position_list = positions_consider
                    designated_positions = selected_positions
                    
                    for sublist in designated_positions:
                        if sublist in position_list: 
                            position_list.remove(sublist)
                    
                    combinations = position_list #after removing selected regions
                
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)

                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor_del = new_tensor_del.clone()
                    if self.args.domain == 'timefreq':
                        altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,-1)
                        altered_tensor_del = altered_tensor_del.reshape(altered_tensor_del.shape[0],1,-1)
                        masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,-1)
                        spectrogram = spectrogram.reshape(spectrogram.shape[0],1,-1)
                    for j in [combinations[a] for a in indices_list[i]]:
                        if self.args.domain == 'time':
                            altered_tensor[:,:,j] = data[:,:,j]
                            altered_tensor_del[:,:,j] = masked_tensor[:,:,j]
                        elif self.args.domain == 'freq':
                            mid_index = freq_output.shape[-1] // 2
                            altered_tensor[:,:,j] = freq_output[:,:,j]
                            altered_tensor_del[:,:,j] = masked_tensor[:,:,j]
                            symmetrical_index = mid_index + (mid_index - j[0]) 
                            altered_tensor[:, :, symmetrical_index] = freq_output[:, :, symmetrical_index]
                            altered_tensor_del[:, :, symmetrical_index] = masked_tensor[:, :, symmetrical_index]
                        else:
                            altered_tensor[:,:,j] = spectrogram[:,:,j]
                            altered_tensor_del[:,:,j] = masked_tensor[:,:,j]           
                    if self.args.domain == 'timefreq':   
                        altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)
                        spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.args.num_freq,self.args.num_slices)
                        altered_tensor_del = altered_tensor_del.reshape(altered_tensor_del.shape[0],1,self.args.num_freq,self.args.num_slices)
                        masked_tensor = masked_tensor.reshape(masked_tensor.shape[0],1,self.args.num_freq,self.args.num_slices)

                    x = altered_tensor.to(device)
                    x_del = altered_tensor_del.to(device) 

                    #apply inverse stft 
                    if self.args.domain == 'timefreq':
                        _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x = torch.tensor(inverse_stft).to(device)
                        _,inverse_stft_del = istft(x_del.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                        x_del = torch.tensor(inverse_stft_del).to(device)
                    elif self.args.domain == 'freq':
                        inverse_fft = ifft(x.cpu().detach().numpy())
                        inverse_fft = np.real(inverse_fft)
                        x = torch.tensor(inverse_fft).to(device)

                        inverse_fft_del = ifft(x_del.cpu().detach().numpy())
                        inverse_fft_del = np.real(inverse_fft_del)
                        x_del= torch.tensor(inverse_fft_del).to(device)
                    
                    #apply forward pass
                    with torch.no_grad(): 
                        x = self.model(x.float())
                        x_del = self.model(x_del.float())
                    
                    x = torch.softmax(x, dim=1)
                    # x = x - initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])
                    x_del = torch.softmax(x_del, dim=1)
                    class_probability_scores_del[i] = torch.sum(x_del[:,self.label])

            total_count += data.shape[0]

        class_probability_scores = [x / total_count for x in class_probability_scores]
        class_probability_scores_del = [x / total_count for x in class_probability_scores_del]

        count = Counter([index for sublist in indices_list for index in sublist])
        summed_scores_for_indices = sum_scores_for_each_index(indices_list, class_probability_scores)
        summed_scores_for_indices_del = sum_scores_for_each_index(indices_list, class_probability_scores_del)
        for i in range(len(summed_scores_for_indices)):
            indices_probability_scores.append(summed_scores_for_indices[i] / count[i])
        for i in range(len(summed_scores_for_indices_del)):
            indices_probability_scores_del.append(summed_scores_for_indices_del[i] / count[i])
        max_position = torch.argmax(self.args.insertion_weight * torch.tensor(indices_probability_scores) - self.args.deletion_weight * torch.tensor(indices_probability_scores_del))
        max_position = combinations[max_position]

        selected_positions.append(max_position)
        return selected_positions, positions_consider            
                
                
             
        
        
        
        
        
    








