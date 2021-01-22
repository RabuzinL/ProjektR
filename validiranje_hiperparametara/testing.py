import matplotlib.pyplot as plt
import numpy as np
import data_load
import torch
import torchvision
from conv import ConvolutionalModel
import data_load


def testing():
    PATH = './CIFAR_10/cifar_netAdam.pth'

    net = ConvolutionalModel(3, 16, 128, 10)
    net.load_state_dict(torch.load(PATH))
    total = 0
    correct = 0
    confMatrix = np.zeros((10, 10), int)
    with torch.no_grad():
        for data in data_load.testloader:
            images, labels = data

            output = net.forward(images)
            _, predictions = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            for i in range(labels.size(0)):
                confMatrix[predictions[i], labels[i]] += 1

    print("Accuracy of the neural network on CIFAR_10 testing set is: %.2f %%" %((correct/total)*100))
    print(data_load.classes)
    print(confMatrix)
    print(total)
    print(correct)
    specificMetrics(confMatrix)

def specificMetrics(confMatrix):
    for i in range(np.size(confMatrix, 0)):
        print("Class: " + data_load.classes[i], end='')
        precc = 0
        recal = 0
        tp = 0
        fp = 0
        fn = 0
        for j in range(np.size(confMatrix, 0)):
            if i == j:
                tp += confMatrix[i, j]
            else:
                fn += confMatrix[j, i]
                fp += confMatrix[i, j]
            
        precc = tp/(tp + fp)
        recal = tp/(tp + fn)

        print(", Precision: %.2f, Recall: %.2f" %(precc, recal))

testing()