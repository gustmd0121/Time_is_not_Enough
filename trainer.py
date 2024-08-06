import os 
import torch 
from torch import nn 
from torch.utils.data import DataLoader, random_split
import torch.optim as optim 
import random 
from data_loader import load_data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import seaborn as sns
from models.models import *
from models.spectralx import *
from metrics.evaluations import *
import argparse
import copy 
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_num_threads(32)
random.seed(42)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassifierTrainer():
    def __init__(self, args):
        self.args = args
        self.savedir = os.path.join(*[self.args.savedir, self.args.dataset, self.args.model_type])
        self.dataset = self.args.dataset
        directory = self.savedir
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Initialize and seed the generator
        self.generator = torch.Generator()
        self.generator.manual_seed(911)

        ds, num_freq, num_slices = load_data(self.dataset, self.args)
        train_size = int(0.8 * len(ds))
        val_size = int(0.1 * len(ds))
        test_size = len(ds) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size], generator=self.generator)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
    
    def train(self, net):
        self.model = net.to(device)
        optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_loss = 1000
        best_model = None
        total_val_loss = []

        #Traning the Network  
        for epoch in range(self.args.n_epochs):
            self.model.train()
            
            training_loss = 0
            
            for _, (data, labels, _, _, _) in enumerate(self.train_loader):
                data = data.unsqueeze(1).float()
                labels = labels.long()
            
                data, labels = data.to(device), labels.to(device)
                output= self.model(data)
            
                loss = criterion(output, labels) 
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item() * data.size(0)
                
            training_loss = training_loss / len(self.train_loader.sampler)
            wandb.log({"Training loss": training_loss})
            
            self.model.eval()
            validation_loss = 0
            for _, (data, labels, _, _, _) in enumerate(self.val_loader):
                data = data.unsqueeze(1).float()
                labels = labels.long()
                data, labels = data.to(device), labels.to(device)
                output= self.model(data) 
                loss = criterion(output, labels) 
                
            validation_loss += loss.item() * data.size(0)
            validation_loss = validation_loss / len(self.val_loader.sampler)
            total_val_loss.append(validation_loss)
            wandb.log({"Validation loss": validation_loss})
            if epoch % 10 == 0:
                print(epoch, validation_loss)
            if best_val_loss > total_val_loss[-1]:
                best_val_loss = total_val_loss[-1]
                best_model = copy.deepcopy(self.model)
                
                savedict = {
                    'args': self.args,
                    'model_state_dict': best_model.state_dict(),
                        }
                
            if epoch % 10 == 0:
                savepath  = os.path.join(self.savedir,f"{self.args.model_type}.pt")
                torch.save(savedict, savepath) 
        
    def test(self, net):
        criterion = torch.nn.CrossEntropyLoss()
        self.model = net.to(device)
        a = torch.load(self.args.classification_model)
        self.model.load_state_dict(a['model_state_dict'])

        for param in self.model.parameters():
            param.requires_grad = False        

        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for _, (data, labels, _, _, _) in enumerate(self.test_loader):
                data = torch.tensor(data)
                data = data.unsqueeze(1).float()
                labels = labels.type(torch.LongTensor) 
                data, labels = data.to(device), labels.to(device)
                output = self.model(data)

                _, predicted = torch.max(output, 1)
                c = (predicted == labels).squeeze()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate precision, recall, and F1 score for each class
        precision = precision_score(all_labels, all_predictions, average=None)
        recall = recall_score(all_labels, all_predictions, average=None)
        f1 = f1_score(all_labels, all_predictions, average=None)
        accuracy = accuracy_score(all_labels, all_predictions)

        for i in range(self.args.num_classes):
            print(f'Class {i} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1: {f1[i]:.2f}, accuracy: {accuracy:.2f}')

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('--classification_model', type=str, default = '/home/hschung/testing/Time_is_not_Enough/classification_models/arrowhead/resnet/resnet.pt', help='trained classifier model for testing')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])
    parser.add_argument('--dataset', type=str, default='arrowhead', help="Dataset to train on")
    parser.add_argument('--model_type', type=str, default="resnet", choices=['resnet', 'transformer', 'bilstm'])
    parser.add_argument('--savedir', type=str, default="classification_models")
    parser.add_argument('--inplanes', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--task', type=str, default='classification', choices = ['spectralx', 'classification'])
    parser.add_argument('--topk', type=int, default=10)

    #spectrogram
    parser.add_argument('--fs', type=int, default=1)
    parser.add_argument('--noverlap', type=int, default=8)
    parser.add_argument('--nperseg', type=int, default=16)

    #xai 
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--num_perturbations', type=int, default=2000)
    parser.add_argument('--selected_regions', type=int, default=10)
    parser.add_argument('--region_height', type=int, default=1)
    parser.add_argument('--region_width', type=int, default=1)
    parser.add_argument('--method', type=str, default= 'combined', choices = ['insertion', 'deletion', 'combined'])
    parser.add_argument('--deletion_weight', type=int, default= 0.8)
    parser.add_argument('--insertion_weight', type=int, default = 0.2)
    parser.add_argument('--group_length', type=int, default=1)
    
    #metric 
    parser.add_argument('--metric_eval', type=bool, default=False)
    parser.add_argument('--step_size', type=int, default=2, choices = [1,2])
    
    #ranking
    parser.add_argument('--ranking', default =  [[[24], [2], [21], [4], [12], [27], [0], [28]]])

    
    #transformer and bi-lstm
    parser.add_argument('--use_transformer', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--timesteps', type=int, default=1024) #128 140 1639
    
    args = parser.parse_args()
    
    if args.model_type == "resnet":
        if args.use_transformer:
            exit()
        net = resnet34(args, num_classes = args.num_classes).to(device)
    elif args.model_type == 'bilstm':
        net = BiLSTMModel(args, num_classes = args.num_classes).to(device) 
    elif args.model_type == 'transformer':
        net = TransformerModel(args, num_classes = args.num_classes).to(device)     

        
    if args.metric_eval:
        trainer = Metric_Eval(args, net)
        trainer.faithfulness_compute() 

        exit()   
    if args.task == 'classification':
        trainer = ClassifierTrainer(args)
        if args.mode == 'train':
            wandb.init(project="New_Xai", name=f"{args.model_type}_dataset{args.dataset}", reinit=True, config={
                "dataset": args.dataset,
                "model type": args.model_type,
                "batch size": args.batch_size,
                "learning rate": args.lr, 
                "task": args.task,
            })
            trainer.train(net)
        elif args.mode == 'test':
            trainer.test(net)
    elif args.task == 'spectralx':        
        trainer = XAITrainer(args, net)

        #insertion
        selected_positions = []
        positions_consider = []
        for i in range(args.topk+1):
            if args.method == 'insertion':
                selected_positions, positions_consider = trainer.insertion(selected_positions, positions_consider)
                print(f"iteration number {i}")
                print(f"selected positions: {selected_positions}")
            elif args.method == 'deletion':
                selected_positions, positions_consider = trainer.deletion(selected_positions, positions_consider)
                print(f"iteration number {i}")
                print(f"selected positions: {selected_positions}")
            elif args.method == 'combined':
                selected_positions, positions_consider = trainer.combined(selected_positions, positions_consider)
                print(f"iteration number {i}")
                print(f"selected positions: {selected_positions}")
            if len(selected_positions) == 8:
                break

        print(f"Final Selected Positions: {selected_positions}")
    
    print("Finished training / testing")