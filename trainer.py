"""Train models on a given dataset."""
import os
import json
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import params_update

hpc=params_update.hpc

if  hpc:
  device = torch.cuda.current_device()
if not hpc:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


OUTPUT_DIR = 'out'
CHECKPOINT_DIR = 'checkpoint'

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

@dataclass
class LoggingParameters:
    """Data class holding parameters for logging."""
    model_name: str
    dataset_type: str
    optimizer_name: str
    optimizer_params: dict
    optimizer_type : str
    learning_rate : float
    dropout_per : float
    n_layers : int
    batch_size : int
    train_per : float
    test_per : float
    n_epochs : int
    num_exp : int
    loss_f : str


class Trainer:
    """Abstract model trainer on a binary classification task."""
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 scheduler,
                 criterion,
                 cos_f,
                 batch_size: int,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 test_dataset: Dataset,
                 data_type: str,
                 dropout : float,
                 loss_f : str,
                 num_exp : int,
                 fast_checkup=False):

        self.loss_f = loss_f
        self.cos_f = cos_f
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.epoch = 0
        self.fast_checkup = fast_checkup
        self.model.apply(weight_init)
        self.num_exp = num_exp


    def loss_function(self,outputs, targets):
        loss = torch.mean((1-torch.abs(self.criterion(targets,outputs))))
        loss.to(device)
        angle = torch.arccos(1-loss)*180/(np.pi)
        angle.to(device)
        if self.loss_f=='cos':
            return loss, angle
        if self.loss_f == 'angle':
            return angle, angle

        

    def train_one_epoch(self) :
    # def train_one_epoch(self) -> tuple[float, float]:

        """Train the model for a single epoch on the training dataset.
        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        self.model.train()
        total_loss = 0
        total_angle = 0
        avg_loss = 0
        nof_samples = 0
    
        train_dataloader = self.train_dataset
        print_every = int(len(train_dataloader.dataset) / 10)
        # with self.experiment.train():
        for batch_idx, (inputs, targets,index) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs).to(device)
            # outputs = self.model(inputs)
            

            # compute loss
            loss,angle = self.loss_function(outputs, targets)

            # backward
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.5)

            # optimizer step
            self.optimizer.step()
            

            # avg loss
            nof_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0) # Multiply in target size
            total_angle += angle.item() * targets.size(0)
            avg_loss = total_loss / nof_samples
            avg_angle =  total_angle / nof_samples


            if batch_idx % print_every == 0 or \
                    batch_idx == len(train_dataloader) - 1:
                # angle = (torch.arccos(1-torch.tensor(avg_loss)))*180/(np.pi)
                # angle = (torch.arccos(cos))*180/(np.pi)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAngle: {:.6f} degrees'.format(
                self.epoch, batch_idx * len(inputs), len(train_dataloader.dataset),
                100.*batch_idx / len(train_dataloader),
                avg_loss, avg_angle))

        return avg_loss, avg_angle

    def evaluate_model_on_dataloader(
            self, data_type, dataset: torch.utils.data.Dataset) :
        """Evaluate model loss and accuracy for dataset.

        Args:
            dataset: the dataset to evaluate the model on.

        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        self.model.eval()
        dataloader = dataset
        total_loss = 0
        total_angle = 0
        avg_loss = 0
        nof_samples = 0
        print_every = max(int(len(dataloader.dataset) / 10), 1)
        # with self.experiment.test():
        for batch_idx, (inputs, targets,index) in enumerate(dataloader):
            
            with torch.no_grad():
                inputs = inputs.to(device)
                targets = targets.to(device)

                # forward pass
                outputs = self.model(inputs).to(device)

                # compute loss
                loss,angle = self.loss_function(outputs, targets)
                # avg loss
                nof_samples += targets.size(0)
                total_loss += loss.item() * targets.size(0)
                total_angle += angle.item() * targets.size(0)
                avg_loss = total_loss / nof_samples
                avg_angle =  total_angle / nof_samples


            if batch_idx % print_every == 0 or batch_idx == len(dataloader) - 1:
                print(str(data_type), 'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAngle: {:.6f} degrees'.format(
                self.epoch, batch_idx * len(inputs), len(dataloader.dataset),
                100.*batch_idx / len(dataloader),
                avg_loss, avg_angle))
                

        return avg_loss, avg_angle

    def train_no_drop(self):
        """Evaluate the model performance."""
        return self.evaluate_model_on_dataloader('train', self.train_dataset)

    def validate(self):
        """Evaluate the model performance."""
        return self.evaluate_model_on_dataloader('validation', self.validation_dataset)

    def test(self):
        """Test the model performance."""
        return self.evaluate_model_on_dataloader('test', self.test_dataset)

    @staticmethod
    def write_output(logging_parameters: LoggingParameters, data: dict, output_filepath: str):
        """Write logs to json.

        Args:
            logging_parameters: LoggingParameters. Some parameters to log.
            data: dict. Holding a dictionary to dump to the output json.
        """
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # output_filename = f"{logging_parameters.dataset_type}_" \
        #                   f"{logging_parameters.model_name}_" \
        #                   f"{logging_parameters.optimizer_name}.json"

        print(f"Writing output to {output_filepath}")
        # Load output file
        if os.path.exists(output_filepath):
            # pylint: disable=C0103
            with open(output_filepath, 'r', encoding='utf-8') as f:
                all_output_data = json.load(f)
        else:
            all_output_data = []

        # Add new data and write to file
        all_output_data.append(data)
        # pylint: disable=C0103
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)

    def write_to_comet(self, data,experiment):
        experiment.log_metric('train_acc', float(data['train_acc'][-1]), epoch=self.epoch)
        experiment.log_metric('val_acc', float(data['val_acc'][-1]), epoch=self.epoch)
        experiment.log_metric('test_acc',float(data['test_acc'][-1]), epoch=self.epoch)
        experiment.log_metric('train_loss', float(data['train_loss'][-1]), epoch=self.epoch)
        experiment.log_metric('val_loss', float(data['val_loss'][-1]), epoch=self.epoch)
        experiment.log_metric('test_loss', float(data['test_loss'][-1]), epoch=self.epoch)
        experiment.log_metric('lr', float(data['lr'][-1]), epoch=self.epoch)

    def run(self, experiment,epochs,CHECKPOINT_DIR ,exp_name,logging_parameters: LoggingParameters):
        """Train, evaluate and test model on dataset, finally log results."""
        best_valid_acc = 10000
        # exp_name =  'Experiment_'+ str(logging_parameters.num_exp)
        output_filename = exp_name +  '.json'
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        # if os.path.exists(output_filepath):
        #     models_exist = os.listdir(OUTPUT_DIR)
        #     matching = [s for s in models_exist if exp_name in s]
        #     exp_name = exp_name + '_'+ str(len(matching)+1)
        #     output_filename = exp_name + '.json'
        #     output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        experiment.log_parameters(logging_parameters)
        # self.experiment.log_parameters(logging_parameters)
        output_data = {
            "model": logging_parameters.model_name,
            "dataset": logging_parameters.dataset_type,
            "optimizer": {
                "name": logging_parameters.optimizer_name,
                "params": logging_parameters.optimizer_params,
            },
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
            "lr": [],
        }
        best_loss = 10000
        model_filename = exp_name + '.pt'
        checkpoint_filename = os.path.join(CHECKPOINT_DIR, model_filename)
        for self.epoch in range(1, epochs + 1):
            print(f'Epoch {self.epoch}/{epochs}')

            train_loss, train_acc = self.train_one_epoch()
            train_loss, train_acc = self.train_no_drop()
            val_loss, val_acc = self.validate()
            test_loss, test_acc = self.test()
            self.scheduler.step()

            output_data["train_loss"].append(train_loss)
            output_data["train_acc"].append(train_acc)
            output_data["val_loss"].append(val_loss)
            output_data["val_acc"].append(val_acc)
            output_data["test_loss"].append(test_loss)
            output_data["test_acc"].append(test_acc)
            output_data["lr"].append(self.optimizer.param_groups[0]["lr"])

            if val_loss<best_valid_acc:
                best_valid_acc = val_loss

            # Save checkpoint
            if val_acc < best_loss:
                print(f'Saving checkpoint {checkpoint_filename}')
                state = {
                    'model': self.model.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'epoch': self.epoch,
                }
                
                torch.save(state, checkpoint_filename)
                best_loss = val_acc

            if not self.fast_checkup:
                self.write_to_comet(data=output_data,experiment=experiment)
        

        self.write_output(logging_parameters, output_data,output_filepath)
        return best_valid_acc