import os
import numpy as np
import librosa
from tqdm import tqdm  # For progress tracking

        
def extract_features(file_path, target_length=400, n_mfcc=40):
    try:
        # 1. Load audio with fixed parameters
        librosa.core.audio.__RESAMPY__ = False
        audio, sr = librosa.load(file_path, 
                                sr=16000,  # Force sample rate
                                duration=3.0,  # Fixed duration
                                res_type='kaiser_fast')
        
        # 2. Preprocessing
        audio = librosa.effects.preemphasis(audio)  # High-pass filter
        
        # 3. Extract MFCCs with consistent parameters
        n_fft = 512
        hop_length = 256
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr,
                                    n_mfcc=n_mfcc,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
        
        # 4. Time dimension adjustment
        if mfccs.shape[1] > target_length:
            # Random crop for longer sequences
            start = np.random.randint(0, mfccs.shape[1] - target_length)
            mfccs = mfccs[:, start:start+target_length]
        else:
            # Pad with mirrored content for shorter sequences
            pad_width = target_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0,0), (0,pad_width)), 
                           mode='reflect')
        
        # 5. Standardization
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        return mfccs.T  # (time_steps, n_mfcc)
    
    except Exception as e:
        print(f"Error in {file_path}: {str(e)}")
        return None        

def process_audio_files(data_dir, output_file):
    """
    Process all audio files in directory and save features
    """
    features = []
    labels = []
    
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, audio_file)
                if file_path.endswith('.wav'):
                    feature = extract_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(speaker_dir)
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Save features and labels
    np.savez(output_file, features=features, labels=labels_encoded)
    
    # Save label encoder
    with open('/content/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    return features, labels_encoded
    
def process_dataset(data_dir, output_file):
    """
    Process dataset with proper directory structure handling
    """
    features = []
    labels = []
    skipped_files = []
    
    # Supported audio formats
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    # Get all audio files recursively
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                audio_files.append(os.path.join(root, file))
    
    # Process files with progress bar
    for file_path in tqdm(audio_files, desc="Processing audio files"):
        # Extract label from directory structure (assuming format: .../speaker_id/...)
        label = os.path.basename(os.path.dirname(file_path))
        
        # Extract features
        feature = extract_features(file_path)
        
        if feature is not None:
            features.append(feature)
            labels.append(label)
        else:
            skipped_files.append(file_path)
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Save features and labels
    np.savez(output_file, features=features, labels=labels)
    
    # Save skipped files list
    if skipped_files:
        with open('skipped_files.txt', 'w') as f:
            f.write('\n'.join(skipped_files))
    
    print(f"\nProcessed {len(features)} files successfully")
    print(f"Skipped {len(skipped_files)} files due to errors")
    
    return features, labels
   
def prepare_dataset(features, labels, test_size=0.3, random_state=42):
    """
    Split dataset into train and test sets with PyTorch DataLoader
    """
    # Add channel dimension for CNN
    features = np.expand_dims(features, axis=1)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )
    
    # Create datasets
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader