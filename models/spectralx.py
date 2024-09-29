import torch 
import pandas as pd 
import random
from collections import Counter
from data_loader import *
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

        ds, height, width = load_data(self.args.dataset, self.args)
        self.height = height
        self.width = width
        groups = generate_region_groups(self.height, self.width, 1, 1)
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
    
    def insertion(self, selected_positions, positions_consider):
        inverse_data = []
        position_scores = []
        indices_probability_scores = []
        class_probability_scores = list(0. for i in range(self.num_per))
        total_count = 0
        for n, (data, labels, spectrogram, rbp_spec, rbp_signal) in enumerate(self.test_loader):
            #check if prediction and labels match
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float() 
            data, data_check, labels, spectrogram, rbp_spec, rbp_signal= data.to(device), data_check.to(device), labels.to(device), spectrogram.to(device), rbp_spec.to(device), rbp_signal.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)
            
            data = data[(labels == self.label) & (predicted == self.label)]
            spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
            rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
            rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
            labels = labels[(labels == self.label) & (predicted == self.label)]
            
            # Check if any samples remain after filtering
            if data.shape[0] == 0:
                continue  # Skip this batch if no samples remain

            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64)

            masked_tensor = rbp_spec.to(torch.complex64).to(device)
    
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
                    
                    #all combinations of positions to consider 
                    for j in [combinations[a] for a in indices_list[i]]:
                        new_tensor[:, :, j] = spectrogram[:, :, j]

                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.height,self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.height,self.width)
                    
                    x = new_tensor.to(device)

                    #apply inverse stft 
                    _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)
                    
                    #apply forward pass
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    # x = x-initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])

            else:
                #insert selected positions
                new_tensor = masked_tensor.clone()
                for position in selected_positions:
                    data = data.reshape(data.shape[0],1,-1).to(torch.float64)
                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,-1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,-1)
                    new_tensor[:,:,position] = spectrogram[:,:,position]
                    new_tensor = new_tensor.reshape(new_tensor.shape[0],1,self.height,self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.height,self.width)

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

                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0],1,self.height,self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0],1,self.height,self.width)
                        
                    x = altered_tensor.to(device) 

                    _,inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)
                    
                    #apply forward pass
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    # x = x - initial_x
                    class_probability_scores[i] = torch.sum(x[:,self.label])
            
            total_count += data.shape[0]
        
        if total_count == 0:
            print("No more samples for the classifier prediction label that matches the gt label. Recommend to train the classifier model more epochs or use different hyperparameters.")
            return selected_positions, positions_consider
        
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

        for n, (data, labels, spectrogram, rbp_spec, rbp_signal) in enumerate(self.test_loader):
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float() 
            data, data_check, labels, spectrogram, rbp_spec, rbp_signal = data.to(device), data_check.to(device), labels.to(device), spectrogram.to(device), rbp_spec.to(device), rbp_signal.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)
            
            data = data[(labels == self.label) & (predicted == self.label)]
            spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
            rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
            rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
            labels = labels[(labels == self.label) & (predicted == self.label)]
            
            # Check if any samples remain after filtering
            if data.shape[0] == 0:
                continue  # Skip this batch if no samples remain

            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64)
            masked_tensor = rbp_spec.to(torch.complex64)

            if not selected_positions: 
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions) 
                positions_consider = self.groups 
                combinations = positions_consider 
                for i in range(self.num_per):
                    new_tensor = spectrogram.clone()
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    masked_tensor = masked_tensor.view(masked_tensor.shape[0], 1, -1)
                    
                    for j in [combinations[a] for a in indices_list[i]]:
                        new_tensor[:, :, j] = masked_tensor[:, :, j]

                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)
                    
                    x = new_tensor.to(device)

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)
                    
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])

            else:
                new_tensor = spectrogram.clone()
                for position in selected_positions:
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    new_tensor[:,:,position] = masked_tensor[:,:,position]
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)

                if n == 0:
                    position_list = positions_consider
                    designated_positions = selected_positions
                    
                    for sublist in designated_positions:
                        if sublist in position_list: 
                            position_list.remove(sublist)
                    
                    combinations = position_list
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)
                
                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    for j in [combinations[a] for a in indices_list[i]]:
                        altered_tensor[:,:,j] = masked_tensor[:,:,j]
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)
                        
                    x = altered_tensor.to(device) 

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)
                    
                    with torch.no_grad(): 
                        x = self.model(x.float())
                    
                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])
            
            total_count += data.shape[0]
        
        if total_count == 0:
            print("No more samples for the classifier prediction label that matches the gt label. Recommend to train the classifier model more epochs or use different hyperparameters.")
            return selected_positions, positions_consider
          
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

        for n, (data, labels, spectrogram, rbp_spec, rbp_signal) in enumerate(self.test_loader):
            data_check = torch.tensor(data)
            data_check = data_check.unsqueeze(1).float() 
            data, data_check, labels, spectrogram, rbp_spec, rbp_signal = data.to(device), data_check.to(device), labels.to(device), spectrogram.to(device), rbp_spec.to(device), rbp_signal.to(device)
            output_check = self.model(data_check)
            _, predicted = torch.max(output_check, 1)
            
            data = data[(labels == self.label) & (predicted == self.label)]
            spectrogram = spectrogram[(labels == self.label) & (predicted == self.label)]
            rbp_spec = rbp_spec[(labels == self.label) & (predicted == self.label)]
            rbp_signal = rbp_signal[(labels == self.label) & (predicted == self.label)]
            labels = labels[(labels == self.label) & (predicted == self.label)]

            # Check if any samples remain after filtering
            if data.shape[0] == 0:
                continue  # Skip this batch if no samples remain

            spectrogram = spectrogram.unsqueeze(1).to(torch.complex64)
            spectrogram_del = spectrogram.clone()

            masked_tensor = rbp_spec.to(torch.complex64)
            masked_tensor_del = masked_tensor.clone()

            if not selected_positions: 
                indices_list = generate_random_indices(self.num_per, len(self.groups), self.selected_regions) 
                positions_consider = self.groups 
                combinations = positions_consider 
                data = data.view(data.shape[0], 1, -1).to(torch.float64)
                for i in range(self.num_per):
                    new_tensor = masked_tensor.clone()
                    new_tensor_del = spectrogram_del.clone()
                    new_tensor = new_tensor.view(new_tensor.shape[0], 1, -1)
                    spectrogram = spectrogram.view(spectrogram.shape[0], 1, -1)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, -1)
                    masked_tensor_del = masked_tensor_del.reshape(masked_tensor_del.shape[0], 1, -1)
                    
                    for j in [combinations[a] for a in indices_list[i]]:
                        new_tensor[:, :, j] = spectrogram[:, :, j]
                        new_tensor_del[:,:,j] = masked_tensor_del[:,:,j]

                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, self.height, self.width)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, self.height, self.width)
                    masked_tensor_del = masked_tensor_del.reshape(masked_tensor_del.shape[0], 1, self.height, self.width)
                    
                    x = new_tensor.to(device)
                    x_del = new_tensor_del.to(device)

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)        
                    _, inverse_stft_del = istft(x_del.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x_del = torch.tensor(inverse_stft_del).to(device)                

                    with torch.no_grad(): 
                        x = self.model(x.float())
                        x_del = self.model(x_del.float())
                    
                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])
                    x_del = torch.softmax(x_del, dim=1)
                    class_probability_scores_del[i] = torch.sum(x_del[:,self.label])
            
            else:
                new_tensor = masked_tensor.clone()
                new_tensor_del = spectrogram.clone()

                for position in selected_positions:
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, -1)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, -1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    new_tensor[:,:,position] = spectrogram[:,:,position]
                    new_tensor_del[:,:,position] = masked_tensor[:,:,position]
                    new_tensor = new_tensor.reshape(new_tensor.shape[0], 1, self.height, self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, self.height, self.width)
                    new_tensor_del = new_tensor_del.reshape(new_tensor_del.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)                   

                if n == 0:
                    position_list = positions_consider
                    designated_positions = selected_positions
                    
                    for sublist in designated_positions:
                        if sublist in position_list: 
                            position_list.remove(sublist)
                    
                    combinations = position_list
                    indices_list = generate_random_indices(self.num_per, len(positions_consider), self.selected_regions)

                for i in range(self.num_per):
                    altered_tensor = new_tensor.clone()
                    altered_tensor_del = new_tensor_del.clone()
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, -1)
                    altered_tensor_del = altered_tensor_del.reshape(altered_tensor_del.shape[0], 1, -1)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, -1)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, -1)
                    for j in [combinations[a] for a in indices_list[i]]:
                        altered_tensor[:,:,j] = spectrogram[:,:,j]
                        altered_tensor_del[:,:,j] = masked_tensor[:,:,j]           
                    altered_tensor = altered_tensor.reshape(altered_tensor.shape[0], 1, self.height, self.width)
                    spectrogram = spectrogram.reshape(spectrogram.shape[0], 1, self.height, self.width)
                    altered_tensor_del = altered_tensor_del.reshape(altered_tensor_del.shape[0], 1, self.height, self.width)
                    masked_tensor = masked_tensor.reshape(masked_tensor.shape[0], 1, self.height, self.width)

                    x = altered_tensor.to(device)
                    x_del = altered_tensor_del.to(device) 

                    _, inverse_stft = istft(x.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x = torch.tensor(inverse_stft).to(device)
                    _, inverse_stft_del = istft(x_del.cpu().detach().numpy(), fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
                    x_del = torch.tensor(inverse_stft_del).to(device)
                    
                    with torch.no_grad(): 
                        x = self.model(x.float())
                        x_del = self.model(x_del.float())
                    
                    x = torch.softmax(x, dim=1)
                    class_probability_scores[i] = torch.sum(x[:,self.label])
                    x_del = torch.softmax(x_del, dim=1)
                    class_probability_scores_del[i] = torch.sum(x_del[:,self.label])

            total_count += data.shape[0]

        if total_count == 0:
            print("No more samples for the classifier prediction label that matches the gt label. Recommend to train the classifier model more epochs or use different hyperparameters.")
            return selected_positions, positions_consider

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
                
                
             
        
        
        
        
        
    








