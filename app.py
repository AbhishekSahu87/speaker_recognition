# app.py
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import torch
from database import Speaker, RecognitionResult, get_db_session
import pickle
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    from model_training import SpeakerCNN  # Import model architecture
    model = SpeakerCNN(input_shape=(1, 40, 400), num_classes=len(le.classes_))
    model.load_state_dict(torch.load('models/speaker_recognition_cnn.pth', map_location=device))
    model.eval()
    return model

model = load_model()
'''
def extract_features(file_path, max_pad_len=400):
    """Extract MFCC features from audio file"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Padding or truncating
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        return mfccs
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None
        '''
import librosa
import numpy as np

def extract_features(file_path, max_pad_len=400):
    """Extract and process MFCC features with proper configuration"""
    try:
        # 1. Load audio with fixed parameters (MUST match training config)
        audio, sr = librosa.load(file_path,
                                sr=16000,          # Fixed sample rate
                                duration=3.0,      # Fixed duration
                                res_type='kaiser_fast',
                                mono=True)         # Convert to mono

        # 2. Pre-emphasis and noise reduction
        audio = librosa.effects.preemphasis(audio)
        
        # 3. Extract MFCCs with consistent parameters
        n_fft = 512
        hop_length = 256
        mfccs = librosa.feature.mfcc(y=audio, sr=sr,
                                    n_mfcc=40,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
        
        # 4. Add delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        # 5. Time axis normalization
        if mfccs.shape[1] > max_pad_len:
            # Random crop for longer sequences
            start = np.random.randint(0, mfccs.shape[1] - max_pad_len)
            mfccs = mfccs[:, start:start+max_pad_len]
        else:
            # Circular padding for shorter sequences
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0,0), (0,pad_width)), mode='wrap')
        
        # 6. Feature normalization (using training stats)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        # 7. Reshape for model input
        return mfccs.T  # Transpose to (time_steps, features)
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None         

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    session = get_db_session()
    try: 
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file:
            # Save uploaded file
            filename = str(uuid.uuid4()) + '.wav'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract features
            features = extract_features(filepath)
            if features is None:
                return jsonify({'error': 'Error processing audio file'})
            
            # Convert to PyTorch tensor and add dimensions
            features_tensor = torch.tensor(features[np.newaxis, np.newaxis, ...], 
                                         dtype=torch.float32).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(features_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            speaker = le.inverse_transform([predicted_idx.item()])[0]
            confidence = confidence.item() * 100
            print("confidence",confidence,"speaker",speaker,"filepath",filepath)
            # Save recognition result to database
            session = get_db_session()
            speaker_record = session.query(Speaker).filter_by(name=speaker).first()
            
            result = RecognitionResult(
                audio_file=filename,
                predicted_speaker_id=speaker_record.id if speaker_record else None,
                confidence=float(confidence)
            )
            session.add(result)
            session.commit()
            
            # Refresh the result to ensure ID is loaded
            session.refresh(result)
            return jsonify({
                'speaker': speaker,
                'confidence': f"{confidence:.2f}%",
                'audio_file': filename,
                'result_id': result.id
            })
    finally:
        session.close()  #

@app.route('/speakers', methods=['GET'])
def list_speakers():
    session = get_db_session()
    speakers = session.query(Speaker).all()
    session.close()
    return jsonify([{
        'id': s.id,
        'name': s.name,
        'sample_count': s.voice_sample_count,
        'created_at': s.created_at.isoformat()
    } for s in speakers])

@app.route('/results', methods=['GET'])
def list_results():
    session = get_db_session()
    results = session.query(RecognitionResult).order_by(RecognitionResult.timestamp.desc()).limit(50).all()
    session.close()
    return jsonify([{
        'id': r.id,
        'audio_file': r.audio_file,
        'predicted_speaker': get_speaker_name(r.predicted_speaker_id),
        'confidence': r.confidence,
        'timestamp': r.timestamp.isoformat()
    } for r in results])

def get_speaker_name(speaker_id):
    session = get_db_session()
    try:
        if not speaker_id:
            return "Unknown"
        speaker = session.query(Speaker).get(speaker_id)
        return speaker.name if speaker else "Deleted Speaker"
    finally:
        session.close()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)