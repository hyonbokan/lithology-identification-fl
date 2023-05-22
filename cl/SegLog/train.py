import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from model import SegLog

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    return train_loss

# Define the main training loop
def main():
    # Set the paths to train and validation directories
    train_dir = 'path_to_train_directory'
    val_dir = 'path_to_validation_directory'

    # Get the list of input files in the train directory
    train_files = [f for f in os.listdir(os.path.join(train_dir, 'x')) if f.endswith('.npy')]

    # Split train files into training and validation sets
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    
    # Create DataLoader for training and validation sets
    train_dataset = CustomDataset(train_files, train_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = CustomDataset(val_files, val_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create SegLog model
    in_channels = 1  # Update with the appropriate number of input channels
    out_channels = 10  # Update with the appropriate number of output channels
    model = SegLog(in_channels, out_channels).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10  # Update with the desired number of training epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'path_to_save_model.pt')

# Execute the training script
if __name__ == '__main__':
    main()
