# Speaker Recognition System
Speaker Recognition System - Develop a noise-resistant speaker recognition system using a CNN-Attention model that accurately identifies
speakers in real-time from short audio clips, maintains greater than 75% accuracy in noisy environments,
and generalizes to new speakers with minimal training data. Combines MFCC feature extraction,
deep learning, and explainable AI for security and voice assistant applications.

# Setting Up the Speaker Recognition System on a Windows/Linux Laptop

Here's a complete step-by-step guide to set up the Speaker Recognition System:

---

### **1. Prerequisites**
- **Git**: [Install Git](https://git-scm.com/downloads)
- **Python 3.8+**: [Install Python](https://www.python.org/downloads/)
- **CUDA Toolkit** (Optional for GPU): [Install CUDA](https://developer.nvidia.com/cuda-downloads)

---

### **2. Clone Repository**
```bash
git clone https://github.com/AbhishekSahu87/speaker_recognition.git
cd speaker_recognition
```
---
This folder structure and files should be there if not created it 
```
speaker_recognition/
├── data/
│   ├── raw/                # Raw audio files
│   ├── processed/          # Processed features
│   └── splits/             # Train/test splits
├── models/                 # Saved models
├── static/                 # Web app static files
├── templates/              # Web app HTML templates
├── feature_extraction.py   # Feature extraction code
├── model_training.py       # Model training code
├── evaluation.py           # Evaluation code
├── app.py                  # Flask web application
└── requirements.txt        # Project dependencies
```

### **3. Set Up Virtual Environment**
#### **Using venv**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### **Using conda**:
```bash
conda create -n speaker python=3.9
conda activate speaker
```

---

### **4. Install Dependencies**
```bash
pip install -r requirements.txt
```

**Note**: For GPU support, first install appropriate PyTorch version from [pytorch.org](https://pytorch.org)

---

### **5. Dataset Preparation**
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset)
2. Organize files:
   ```
   speaker-recognition-system/
   └── data/
       └── raw/
           ├── Speaker_0001/
           │   ├── file1.wav
           │   └── file2.wav
           ├── Speaker_0002/
           └── ...
   ```

---

### **6. Database Setup**
```bash
python -c "from database import init_db; init_db()"
```

---

### **7. Train the Model**
```bash
python main.py
```

**Optional Parameters**:
```bash
python main.py --epochs 100 --batch_size 64 --model_name speaker_cnn
```

---

### **8. Test the Model**
```bash
python test.py  # Or use the Jupyter notebook for analysis
```

---

### **9. Run Web Application**
```bash
python app.py
```

For production:
```bash
gunicorn --bind 0.0.0.0:5000 --timeout 600 app:app
```

---

### **10. Access Application**
Open in browser:  
```
http://localhost:5000
```
