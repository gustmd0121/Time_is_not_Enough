import torch 
import random
import numpy as np 

random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_indices(num_lists, max_index, list_length):
    random_lists = []
    for _ in range(num_lists):
        random_lists.append(random.sample(range(max_index), list_length))
    return random_lists

def generate_region_groups(rows, cols, region_height, region_width):
    groups = []

    for i in range(0, rows - region_height + 1, region_height):
        for j in range(0, cols - region_width + 1, region_width):
            group = [
                i * cols + j + x * cols + y
                for x in range(region_height)
                for y in range(region_width)
            ]
            groups.append(group)

    return groups

def sum_scores_for_each_index(indices_list, score_list):
    unique_indices = set(idx for sublist in indices_list for idx in sublist)
    summed_scores = {}
    
    for idx in unique_indices:
        total_score = sum(score_list[i] for i in range(len(indices_list)) if idx in indices_list[i])
        summed_scores[idx] = total_score
    
    return summed_scores

def generate_timestep_groups(total_timesteps, group_size):
     groups = [list(range(start, start + group_size)) for start in range(0, total_timesteps, group_size)]
     if total_timesteps in groups[-1]:
         groups = groups[:-1]
     return groups 

def generate_frequency_groups(total_timesteps, group_size):
     groups = [list(range(start, start + group_size)) for start in range(1, total_timesteps//2, group_size)]
     if total_timesteps in groups[-1]:
         groups = groups[:-1]
     return groups 
 
def add_noise(ts):
    mu, sigma = 0, 0.1 # mean and standard deviation
    noise = np.random.normal(mu, sigma, ts.shape[0])
    noisy_ts = np.add(ts.reshape(ts.shape[0]),noise.reshape(ts.shape[0]))
    return noisy_ts