# Autoencoder

This repository contains an implementation of an autoencoder using PyTorch. The autoencoder is trained on the MNIST dataset to reconstruct the input images.

## Model Architecture
The autoencoder model consists of an encoder and a decoder. The encoder takes an input image and maps it to a lower-dimensional representation, and the decoder reconstructs the image from the encoded representation. The architecture of the model is as follows:

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

## Training
The autoencoder model is trained using the Mean Squared Error (MSE) loss function and optimized with the Adam optimizer. The training process involves iterating over the MNIST dataset for a specified number of epochs.

```python
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        target=torch.tensor(target, dtype=torch.float32).view(data.size(0), -1)
        outputs = model(target)
        loss = criterion(outputs, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
```

## Results
During the training process, the loss is printed at regular intervals to monitor the progress of the model. After training, the autoencoder should be able to reconstruct the input images with minimal loss.

Please refer to the code files for more details.
