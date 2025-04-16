# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import pickle

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

        
class SpeakerCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # input_shape: (time_steps, n_mfcc)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2,2)),
            
            nn.Conv2d(64, 128, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2,2)),
            
            nn.Conv2d(128, 256, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((None, 1))  # Preserve time dimension
        )
        
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, time_steps, n_mfcc)
        x = self.conv_block(x)  # (batch, 256, time_steps//4, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (batch, time_steps//4, 256)
        
        # Temporal attention
        attn_weights = self.attention(x)
        x = torch.sum(x * attn_weights, dim=1)
        
        return self.classifier(x)        

class SpeakerLSTM(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SpeakerLSTM, self).__init__()
        # Calculate input_size based on input_shape
        input_size = input_shape[1] * input_shape[2]  # input_shape[1]: 40, input_shape[2]: 400
        self.lstm = nn.LSTM(
            input_size=input_size, # Changed to calculated input_size
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Ensure x is properly reshaped
        x = x.view(batch_size, -1, self.lstm.input_size) # Changed to self.lstm.input_size
        output, (hn, cn) = self.lstm(x)
        return self.classifier(hn[-1])

def train_model(model, train_loader, val_loader, model_name, num_classes, 
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
               epochs=50, lr=0.001):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    #history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    history = {'history': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Save history
        history['history']['train_loss'].append(avg_train_loss)
        history['history']['val_loss'].append(avg_val_loss)
        history['history']['train_acc'].append(train_acc)
        history['history']['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}\n')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
    
    # Save final model
    torch.save(model.state_dict(), f'models/{model_name}_final.pth')
    return history

def evaluate_model(model, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, np.array(all_preds), np.array(all_labels)