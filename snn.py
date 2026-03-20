
import torch as torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
class LIFNeuron(nn.Module):

    def __init__(self, threshold, beta, surrogate_alpha=2.0):
        super(LIFNeuron, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.membrane_potential = None
        self.surrogate_alpha = surrogate_alpha
        
    def forward(self, x):
        # verifying that self.membrane_potential is not none
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x)
        # update the membrane potential, using V[t] = β · V[t-1] + x
        self.membrane_potential = self.beta * self.membrane_potential + x
        # creating temp variable to store the result of the comparison.
        membrane_temp = SGF.apply(self.membrane_potential - self.threshold, self.surrogate_alpha)
        self.membrane_potential = self.membrane_potential - self.threshold * membrane_temp

        return membrane_temp
    def reset(self):
        self.membrane_potential = None


# Surrogate Gradient Function (SGF) for backpropagation through the LIF neuron
class SGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        temp = (x >= 0).float()
        return temp

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        surrogate = 1 / (torch.pi * (1 + (torch.pi * alpha * x) ** 2))
        return grad_output * surrogate, None
    
class SNNModel(nn.Module):
    def __init__(self, threshold=1.0, beta=0.9, num_classes=2):
        super(SNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.lif1 = LIFNeuron(threshold, beta)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif2 = LIFNeuron(threshold, beta)
        self.bn2 = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.lif3 = LIFNeuron(threshold, beta)
        self.bn3 = nn.BatchNorm2d(256)
        self.avgpool2 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*8*8, num_classes)
        self.lif4 = LIFNeuron(threshold, beta)
        self.threshold = threshold
        self.beta = beta

    # expand the output layer to accommodate new classes while retaining the weights of existing classes

    def expand_output(self, classes):
        temp_fc = nn.Linear(256*8*8, classes) 
        device = self.fc.weight.device
        temp_fc.weight.data[:self.fc.out_features] = self.fc.weight.data
        temp_fc.bias.data[:self.fc.out_features] = self.fc.bias.data
        self.fc = temp_fc
        self.fc.to(device)
        self.lif4 = LIFNeuron(self.threshold, self.beta)
        self.lif4.to(device)
        

        
    


    def forward(self, x):
        self.reset()  # Reset the state of LIF neurons at the beginning of each forward pass
        accumulator = 0
        for t in range(10):  # Simulate for 10 time steps
            output = self.conv1(x)
            output = self.bn1(output)
            output = self.lif1(output)
            output = self.conv2(output)
            output = self.bn2(output)
            output = self.avgpool(output)
            output = self.lif2(output)
            output = self.conv3(output)
            output = self.bn3(output)
            output = self.lif3(output)
            output = self.avgpool2(output)
            output = self.flatten(output)
            output = self.fc(output)
            output = self.lif4(output)
            accumulator += output
        org_x = accumulator / 10
        return org_x
    

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()
        self.lif3.reset()
        self.lif4.reset()
    

class dataLoader:
    def __init__(self, batch_size=32, data_path="./data"):
        self.batch_size = batch_size
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def get_train_loader(self):
        train_dataset = tv.datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def get_test_loader(self):
        test_dataset = tv.datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
    

class trainer:
    def __init__(self, model, cil_manager, lr=0.001, num_epochs=2):
        self.model = model
        self.cil_manager = cil_manager
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_2_epoch(self):
        self.model.train()
        for i in range(len(self.cil_manager.task_split)):
            if i > 0:
                self.model.expand_output((2*i)+2)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                total_loss = 0
            training_data = self.cil_manager.get_task_train_loader(i)
            total_loss = 0
            print(f"Training on Task {i+1} with classes: {self.cil_manager.task_split[i]}")
            for epoch in range(self.num_epochs):
                for batch, (data, target) in enumerate(training_data):
                    data, target = data.to(self.model.fc.weight.device), target.to(self.model.fc.weight.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    total_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    if batch % 100 == 0:
                        print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch+1} completed. average loss: {total_loss / len(self.cil_manager.task_split):.4f}")
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        data_loader = self.cil_manager.get_task_test_loader()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.model.fc.weight.device), target.to(self.model.fc.weight.device) 
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f"Accuracy: {100 * correct / total:.2f}%")


class CILManager:
    def __init__(self, test_data, task_split, train_data, batch_size, data_path="./data"):
        self.test_data = test_data
        self.train_data = train_data
        self.batch_size = batch_size
        self.task_split = task_split
        self.classes_so_far = []
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    def get_task_train_loader(self, task_num):
        indices = []
        classes = self.task_split[task_num]
        for i, label in enumerate(self.train_data.targets):
            if label in classes:
                indices.append(i)
        self.classes_so_far.extend(classes)

        task_train_dataset = torch.utils.data.Subset(self.train_data, indices)
        task_train_loader = torch.utils.data.DataLoader(task_train_dataset, batch_size=self.batch_size, shuffle=True)
        return task_train_loader
    
    def get_task_test_loader(self):
        indices = []

        for i, label in enumerate(self.test_data.targets):
            if label in self.classes_so_far:
                indices.append(i)

        task_test_dataset = torch.utils.data.Subset(self.test_data, indices)
        task_test_loader = torch.utils.data.DataLoader(task_test_dataset, batch_size=self.batch_size, shuffle=False)
        return task_test_loader

if __name__ == "__main__":
    data = dataLoader(batch_size=64)
    train_data = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=data.transform)
    test_data = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=data.transform)
    model = SNNModel()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    task_split = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    cil_manager = CILManager(test_data, task_split, train_data, batch_size=64)
    train = trainer(model, cil_manager, lr=0.001, num_epochs=2)
    train.train_2_epoch()