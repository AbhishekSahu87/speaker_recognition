# evaluation.py
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

def add_noise(audio, noise_level=0.005):
    """
    Add random noise to audio
    """
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def test_noise_robustness(model, X_test, y_test, label_encoder, noise_levels):
    """
    Test model performance at different noise levels
    """
    results = {}
    
    for level in noise_levels:
        # Add noise to test set
        X_test_noisy = np.array([add_noise(x, level) for x in X_test])
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test_noisy, y_test, verbose=0)
        results[level] = accuracy
        
        # Print classification report at highest noise level
        if level == max(noise_levels):
            y_pred = model.predict(X_test_noisy)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            print(f"\nClassification Report at Noise Level {level}:")
            print(classification_report(y_test, y_pred_classes, 
                                      target_names=label_encoder.classes_))
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', 
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_)
            plt.title(f'Confusion Matrix at Noise Level {level}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'static/confusion_matrix_noise_{level}.png')
            plt.close()
    
    # Plot accuracy vs noise level
    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), 'bo-')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy at Different Noise Levels')
    plt.grid(True)
    plt.savefig('static/noise_robustness.png')
    plt.close()
    
    return results

def plot_training_history(history, model_name):
    """
    Plot training and validation accuracy/loss
    """
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'static/{model_name}_training_history.png')
    plt.close()