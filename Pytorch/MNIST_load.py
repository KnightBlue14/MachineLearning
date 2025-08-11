import torch
from torch import nn

import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):


    #    Building blocks of convolutional neural network.

    #    Parameters:
    #        * in_channels: Number of channels in the input image (for grayscale images, 1)

       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
    
    def forward(self, x):

        #    Define the forward pass of the neural network.

        #    Parameters:
        #        x: Input tensor.

        #    Returns:
        #        torch.Tensor
        #            The output tensor after passing through the network.

       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x

# Create a new model
loaded_model = CNN(in_channels=1, num_classes=10)

# Load the saved model
loaded_model.load_state_dict(torch.load('MNISTmodel.pth'))
print(loaded_model)