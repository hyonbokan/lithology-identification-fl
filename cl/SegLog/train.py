import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from model import SegLog
from loader import SegLogDataset


# Set DEVICE (GPU if available, else CPU)
DEVICE = torch.DEVICE('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = 14 # number of curves
out_channels = 13 # number of classes
BATCH = 1

# Set the paths to train and validation directories
train_x_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog/train/x'
train_y_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog/train/y'
val_x_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog/val/x'
val_y_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog/val/y'


# Define the training function
def train(model, train_loader, criterion, optimizer, DEVICE):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    return train_loss, train_acc


# Define the main training loop
def main():

    train_dataset = SegLogDataset(train_x_dir, train_y_dir)
    val_dataset = SegLogDataset(val_x_dir, val_y_dir) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    model = SegLog(in_channels, out_channels).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10  # Update with the desired number of training epochs
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100.0 * correct / total
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

    
    # Save the trained model
    torch.save(model.state_dict(), 'path_to_save_model.pt')

# Execute the training script
if __name__ == '__main__':
    main()
