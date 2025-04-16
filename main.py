# main.py
import numpy as np
from feature_extraction import process_audio_files, prepare_dataset
from model_training import SpeakerCNN, SpeakerLSTM, train_model, evaluate_model
from evaluation import plot_training_history, test_noise_robustness
import pickle
import torch

def main():
    # Step 1: Feature extraction
    print("Extracting features from audio files...")
    features, labels = process_audio_files('data/raw', 'data/processed/features.npz')
    
    # Step 2: Prepare dataset
    print("Preparing train/test splits...")
    train_loader, test_loader = prepare_dataset(features, labels)
    
    # Load label encoder
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 3: Build and train CNN model
    print("Building CNN model...")
    cnn_model = SpeakerCNN(input_shape=(1, 40, 400), num_classes=len(le.classes_)).to(device)
    cnn_history = train_model(cnn_model, train_loader, test_loader, 
                            'speaker_recognition_cnn', len(le.classes_), 
                            device=device, epochs=50)
    
    # Evaluate CNN model
    cnn_accuracy, _, y_test = evaluate_model(cnn_model, test_loader, device)
    print(f"CNN Model Test Accuracy: {cnn_accuracy:.2%}")
    
    # Plot training history
    plot_training_history(cnn_history, 'speaker_recognition_cnn')
    '''
    # Step 4: Build and train LSTM model
    print("\nBuilding LSTM model...")
    lstm_model = SpeakerLSTM(input_shape=(1, 40, 400), num_classes=len(le.classes_)).to(device)
    lstm_history = train_model(lstm_model, train_loader, test_loader, 
                             'speaker_recognition_lstm', len(le.classes_), 
                             device=device, epochs=30)
    
    # Evaluate LSTM model
    lstm_accuracy, _, _ = evaluate_model(lstm_model, test_loader, device)
    print(f"LSTM Model Test Accuracy: {lstm_accuracy:.2%}")
    
    # Plot training history
    plot_training_history(lstm_history, 'speaker_recognition_lstm')
    '''
    print("\nTraining complete. Models saved in 'models/' directory.")

if __name__ == '__main__':
    main()