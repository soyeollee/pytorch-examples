import numpy as np
from torchvision import datasets

trainset = datasets.FashionMNIST('./', download=True, train=True)
testset = datasets.FashionMNIST('./', download=True, train=False)

train_images = trainset.train_data.numpy()
train_labels = trainset.train_labels.numpy()

test_images = testset.test_data.numpy()
test_labels = testset.test_labels.numpy()

np.savez('../data/fashion-mnist.npz', x_train=train_images, y_train=train_labels, x_test=test_images, y_test=test_labels)

