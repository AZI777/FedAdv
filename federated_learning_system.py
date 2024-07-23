import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import random
from model import optimizer_fn, criterion

# 检查CUDA是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 4096
EPOCHS = 1
n_clients = 5
fraction_client = 1.0
ALPHA = 1.0


def train(model, dataloader, epochs):
    model.train()
    optimizer = optimizer_fn(model.parameters())
    train_loss = 0
    n_input = 0
    correct = 0
    for _ in range(epochs):
        for data, target in dataloader:
            n_input += len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.to(device)
            loss = criterion(output, target).to(device)
            train_loss += loss.item() * len(data)
            pred = output.argmax(dim=1)
            correct += torch.sum(pred == target).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    average_correct = correct / n_input
    average_loss = train_loss / n_input
    return average_loss, average_correct


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    n_input = 0
    evaluate_loss = 0.
    with torch.no_grad():
        for data, target in dataloader:
            n_input += len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.to(device)
            loss = criterion(output, target).to(device)
            evaluate_loss += loss.item() * len(data)
            pred = output.argmax(dim=1)
            correct += torch.sum(pred == target).item()
    average_loss = evaluate_loss / n_input
    average_correct = correct / n_input
    return average_loss, average_correct


class Fl_Model(object):
    def __init__(self, model):
        super(Fl_Model, self).__init__()
        self.model = model.to(device)
        self.weight = [name for name, _ in self.model.state_dict().items()]

    def load_weight(self, epoch):
        self.model.load_state_dict(torch.load('model_epoch_' + str(epoch)+".pth"))


class Client(Fl_Model):
    def __init__(self, model, server, dataset, epochs, batch_size=BATCH_SIZE):
        super(Client, self).__init__(model)
        self.server = server
        self.datasize = len(dataset)
        self.epochs = epochs
        self.dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    def synchronize_from_server(self):
        self.model.load_state_dict(self.server.model.state_dict())

    def train(self):
        average_loss, average_correct = train(self.model, self.dataloader, self.epochs)
        return average_loss, average_correct


class Server(Fl_Model):
    def __init__(self, Model, test_dataset, n_rounds, fraction=fraction_client, *train_datasets):
        super(Server, self).__init__(Model())
        self.n_rounds = n_rounds
        self.clients = []
        self.fraction = fraction
        self.test_dataset = test_dataset
        for dataset in train_datasets:
            self.clients.append(Client(Model(), self, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE))

    def synchronize(self):
        for client in self.clients:
            client.synchronize_from_server()

    def update(self):
        self.selected_index = random.sample(range(len(self.clients)), k=max(int(len(self.clients) * self.fraction), 1))
        train_loss = 0.
        train_input = 0
        train_correct = 0
        selected_clients = [self.clients[index] for index in self.selected_index]
        for client in selected_clients:
            train_input += client.datasize
            # l,c表示average_loss,average_correct
            l, c = client.train()
            train_loss += l * client.datasize
            train_correct += c * client.datasize
        average_loss = train_loss / train_input
        average_correct = train_correct / train_input
        return average_loss, average_correct

    def aggregate(self):
        total_datasize = 0
        selected_clients = [self.clients[index] for index in self.selected_index]
        for client in selected_clients:
            total_datasize += client.datasize

        name_list = [key for key, _ in self.model.state_dict().items()]
        init_state_dict = {}
        for name in name_list:
            weight = torch.zeros_like(self.clients[self.selected_index[0]].model.state_dict()[name], dtype=torch.float)
            for client in selected_clients:
                ratio = client.datasize / total_datasize
                weight.add_(client.model.state_dict()[name].float() * ratio)
            init_state_dict[name] = weight
        self.model.load_state_dict(init_state_dict)

    def train(self):
        self.synchronize()
        loss, correct = self.update()
        self.aggregate()
        return loss, correct

    def evaluate(self):
        dataloader = DataLoader(self.test_dataset, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)
        loss, accuracy = evaluate(self.model, dataloader)
        return loss, accuracy

    def run(self, train_loss_list, train_accuracy_list, evaluate_loss_list, evaluate_accuracy_list, start=1):
        if start >= 10:
            start = start - start % 10
            self.load_weight(start)
        else:
            start = 0

        for i in range(start + 1, self.n_rounds + 1):
            train_loss, train_accuracy = self.train()
            evaluate_loss, evaluate_accuracy = self.evaluate()
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            evaluate_loss_list.append(evaluate_loss)
            evaluate_accuracy_list.append(evaluate_accuracy)

            if i % 10 == 0:
                torch.save(self.model.state_dict(), f'model_epoch_{i}.pth')

            print("Round: {}".format(i))
            print(f"Train_loss: {train_loss:.4f}   Train_accuracy: {(100. * train_accuracy):.4f}%")
            print(f"Evaluate_loss: {evaluate_loss:.4f}   Evaluate_accuracy: {(100. * evaluate_accuracy):.4f}%")
            print("")
            with open('loss_acc.log', mode='a') as f:
                f.write("Round: {}\n".format(i))
                f.write(f"Train_loss: {train_loss:.4f}   Train_accuracy: {(100. * train_accuracy):.4f}%\n")
                f.write(f"Evaluate_loss: {evaluate_loss:.4f}   Evaluate_accuracy: {(100. * evaluate_accuracy):.4f}%\n")
                f.write("\n")
