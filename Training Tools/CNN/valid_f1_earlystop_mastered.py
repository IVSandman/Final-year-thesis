#After fix log
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, SubsetRandomSampler
import numpy as np
import torchvision.models as models
from datetime import datetime
import os

# Custom Dataset to Load npz Data
class DNASequencesDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.dna_sequences = data['dna_sequences']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.dna_sequences)
    
    def __getitem__(self, idx):
        dna_seq = torch.tensor(self.dna_sequences[idx], dtype=torch.float32)  # [4, 224, 224]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return dna_seq, label

# Model Definition
class VGG16Modified(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16Modified, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        
        # Modify input layer to 4 channels
        vgg16.features[0] = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Retain the rest of the model
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  # Modify to match the number of output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.classifier(x)
        return x

# Create log file and log messages
def create_log(npz_file, checkpoint_interval, num_epochs, log_dir, learning_rate, batch_size, test_size, val_size, num_classes, optimizer):
    current_date = datetime.now().strftime("%d-%m-%y")
    
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, f'log_{current_date}.txt')
    
    with open(log_filename, 'w') as log_file:
        log_file.write(f"File: {npz_file}\n")
        log_file.write(f"Running date: {current_date}\n")
        log_file.write("Hyperparameters:\n")
        log_file.write(f" - Number of epochs: {num_epochs}\n")
        log_file.write(f" - Checkpoint interval: {checkpoint_interval}\n")
        log_file.write(f" - Learning rate: {learning_rate}\n")
        log_file.write(f" - Batch size: {batch_size}\n")
        log_file.write(f" - Test size: {test_size}\n")
        log_file.write(f" - Validation size: {val_size}\n")
        log_file.write(f" - Number of classes: {num_classes}\n")
        
        # Log optimizer details
        log_file.write("Optimizer:\n")
        log_file.write(f" - Optimizer Type: {optimizer.__class__.__name__}\n")
        log_file.write(f" - Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        
        # Add specific optimizer details if available
        if isinstance(optimizer, optim.SGD):
            log_file.write(f" - Momentum: {optimizer.param_groups[0].get('momentum', 0)}\n")
            log_file.write(f" - Weight Decay: {optimizer.param_groups[0].get('weight_decay', 0)}\n")
        elif isinstance(optimizer, optim.Adam):
            log_file.write(f" - Betas: {optimizer.param_groups[0].get('betas', (0.9, 0.999))}\n")
            log_file.write(f" - Weight Decay: {optimizer.param_groups[0].get('weight_decay', 0)}\n")

        log_file.write(f"Log directory: {log_dir}\n")
        
    return log_filename

def log_epoch(log_filename, epoch, test_loss, test_acc, val_loss, val_acc):
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Epoch {epoch} | test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}, ")
        log_file.write(f"val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}\n")

# Split data into train, validation, and test sets
def split_data(dataset, test_size=0.2, val_size=0.1, batch_size=32):
    dataset_size = len(dataset)
    
    # Split into train and test sets
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    
    # Further split train set into train and validation sets
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size / (1 - test_size), random_state=42)
    
    # Define samplers for training, validation, and test sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create DataLoaders for each split
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader

from sklearn.metrics import classification_report

def train_model(
    model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, 
    device, checkpoint_interval, log_filename, checkpoint_dir, best_model_path, patience=10
):
    model.to(device)
    best_val_acc = 0.0
    epochs_since_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        # Calculate training accuracy
        train_accuracy = correct_predictions / total_predictions

        # Validate the model and calculate metrics
        val_loss, val_acc, val_f1_report = validate_model(model, val_loader, criterion, device)

        # Evaluate on the test set
        test_loss, test_acc = test_model(model, test_loader, criterion, device, log=False)

        # Save the best model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.4f} at epoch {epoch}")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Log training, validation, and test metrics
        if log_filename:
            with open(log_filename, 'a') as log_file:
                log_file.write(
                    f"Epoch {epoch} | train_loss = {running_loss / len(train_loader):.4f}, "
                    f"train_acc = {train_accuracy:.4f}, "
                    f"val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}, "
                    f"test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}\n"
                )
                log_file.write(f"Classification Report:\n{val_f1_report}\n")

        # Save checkpoints at specified intervals
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, 0, checkpoint_path)

        # Handle early stopping
        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best validation accuracy: {best_val_acc:.4f}")
            break


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_val_loss = 0.0
    correct_val_predictions = 0
    total_val_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_val_predictions += (predicted == labels).sum().item()
            total_val_predictions += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate validation loss and accuracy
    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_val_predictions / total_val_predictions

    # Generate classification report for F1-score
    val_f1_report = classification_report(all_labels, all_predictions, digits=4)

    return val_loss, val_accuracy, val_f1_report


def test_model(model, test_loader, criterion, device, log=True):
    model.eval()  # Set model to evaluation mode
    running_test_loss = 0.0
    correct_test_predictions = 0
    total_test_predictions = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_test_predictions += (predicted == labels).sum().item()
            total_test_predictions += labels.size(0)

    # Calculate test loss and accuracy
    test_loss = running_test_loss / len(test_loader)
    test_accuracy = correct_test_predictions / total_test_predictions

    if log:
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy


# Save checkpoint
def save_checkpoint(model, optimizer, epoch, step, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    # Hyperparameters and configurations
    npz_file = '/home/user/torch_shrimp/until-tools/mod/Shrimp_V1_5.npz'
    num_classes = 3
    learning_rate = 0.0001
    num_epochs = 20
    batch_size = 32
    test_size = 0.2
    val_size = 0.2
    checkpoint_interval = 2
    momentum = 0.9
    weight_decay = 0.001
    patience =  1 
    log_dir = '/home/user/torch_shrimp/until-tools/mod/vgg16_mod/file_tunning/tune_14/tune14_b10'
    checkpoint_dir = log_dir  # Directory for saving checkpoints
    model_save_path = os.path.join(log_dir, 'saved_model.pth')  # Final model save path

    # Create dataset and split into train, val, test DataLoaders
    dataset = DNASequencesDataset(npz_file)
    train_loader, val_loader, test_loader = split_data(dataset, test_size=test_size, val_size=val_size, batch_size=batch_size)
    
    # Initialize the model
    model = VGG16Modified(num_classes=num_classes)
    
    # Define criterion and optimizer with momentum
    criterion = nn.CrossEntropyLoss()

    # Use SGD with momentum instead of Adam
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay = weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create log file
    log_filename = create_log(npz_file, checkpoint_interval, num_epochs, log_dir, learning_rate, batch_size, test_size, val_size, num_classes, optimizer)

    # Training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train model
    train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device, checkpoint_interval, log_filename, checkpoint_dir, model_save_path,patience=patience)


