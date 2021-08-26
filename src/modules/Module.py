import torch
import torch.nn as nn
from tqdm import tqdm

class Module(nn.Module):
    r"""This is the root Module of All module so the Model can train just passing .fit in the training process"""
    def __init__(self):
        super(Module, self).__init__()
        self.model = None

    def fit(self, TrainDataLoader, TestDataLoader, epochs, loss_function, optimizer, scheduler=None):
        r"""
        input:
        TrainDataLoader: train dataset that already process to torch.utils.DataLoader type: torch.utils.DataLoader
        TestDataLoader: test dataset that already process to torch.utils.DataLoader type: torch.utils.DataLoader
        epochs: Number of epochses that you want to in training process type: int
        loss_function: The loss function that you're gonna use in the model
        optimizer: Optimization function that you want to use in the model
        scheduler: is how we schedule the learning rate while training
        
        Process:
        Training the model until fit our problem

        output:
        History of the train loss, train accuracy, test/val loss, test/val accuracy
        type: list of dictionaries -> dict{train_loss, train_acc, val_loss, val_acc}
        """
        history = []
        for epoch in range(epochs):
            dash = "="*10
            print(epoch+1)
            print(dash)

            train_loss, train_acc = self._train(TrainDataLoader, loss_function, optimizer)
            test_loss, test_acc = self.eval(TestDataLoader, loss_function)

            if scheduler is not None:
                scheduler.step(test_loss)
            
            history.append({"train_loss":train_loss, "train_acc":train_acc, "val_loss":val_loss, "val_acc":val_acc})

        return history

        

    def _train(self, TrainDataLoader, loss_function, optimizer):
        r"""
        input:
        TrainDataLoader: train dataset that already process to torch.utils.DataLoader type: torch.utils.DataLoader
        loss_function: The loss function that you're gonna use in the model
        optimizer: Optimization function that you want to use in the model
        
        Process:
        Training single epoch
        output:
        Train loss and Train accuracy on single epoch
        """
        self.model.train()
        fin_accuracy = 0
        fin_loss = 0
        tk0 = tqdm(TrainDataLoader, total=len(TrainDataLoader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to('cuda')
            input_data, label = data.values()
            optimizer.zero_grad()
            output = model(input_data)
            loss = loss_function(output, label)
            accuracy = (label == output).sum()/len(output)
            loss.backward()
            optimizer.step()
            fin_loss += loss.item()
            fin_accuracy += accuracy
        train_loss = fin_loss / len(TrainDataLoader)
        train_acc = fin_accuracy/ len(TrainDataLoader)
        
        print(f"Train Loss = {train_loss}  Train Accuracy = {train_acc}")

        return train_loss, train_acc

    def eval(self,TestDataLoader, loss_function):
        r"""
        input:
        TestDataLoader: test dataset that already process to torch.utils.DataLoader type: torch.utils.DataLoader
        loss_function: The loss function that you're gonna use in the model
        optimizer: Optimization function that you want to use in the model
        
        Process:
        Evaluating the model
        output:
        Test/Val loss and Test/Val accuracy
        """
        self.model.eval()
        fin_accuracy = 0
        fin_loss = 0
        tk0 = tqdm(TestDataLoader, total=len(TestDataLoader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to('cuda')
            input_data, label = data.values()
            output = model(input_data)
            loss = loss_function(output, label)
            accuracy = (label == output).sum()/len(output)
            fin_loss += loss.item()
            fin_accuracy += accuracy
        test_loss = fin_loss / len(TestDataLoader)
        test_acc = fin_accuracy/ len(TestDataLoader)
        
        print(f"Test Loss = {test_loss}  Test Accuracy = {test_acc}")

        return test_loss, test_acc