from conv import ConvolutionalModel
import torch.optim as optim
from torch import nn
import data_load
import torch
import evaluate
import plot

def trainNetwork():
    """Performs a standard procedure for training a neural network.
    Training progress after each learning epoch is evaluated in order to
    gain insigth into ConvNets continuous performance.
    Important notes
    ---------------
    Loss function: Cross entropy loss

    Optimizer: Adam
    
    Scheduler: ExponentialLR
    """
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []
    SAVE_DIR = 'C:/Users/Dario/Desktop/Projekt/Drugi labos/lab2_torch/figures'
    device = torch.device('cuda')

    net = ConvolutionalModel(3, 16, 128, 10).to(device=device)

    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    epoch = 10
    

    for e in range(epoch):
    
        accLoss = 0.0

        for i, data in enumerate(data_load.trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()

            outputs = net.forward(inputs)
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()

            accLoss += loss.item()

            if i % 100 == 0:
                print("Epoch: %d, Iteration: %5d, Loss: %.3f" % ((e + 1), (i), (accLoss / (i + 1))))
                
        train_loss, train_acc = evaluate.evaluate(net, False)
        val_loss, val_acc = evaluate.evaluate(net, True)

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        plot_data['lr'] += [scheduler.get_lr()]

        scheduler.step()

    plot.plot_training_progress(SAVE_DIR, plot_data)
    PATH = './CIFAR_10/cifar_netAdam.pth'
    torch.save(net.state_dict(), PATH)


trainNetwork()


