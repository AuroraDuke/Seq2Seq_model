import os
import matplotlib.pyplot as plt

def plot_training_history(history, model_name='model'):
    """
    Eğitim geçmişini görselleştirir ve kaydeder.

    Parametreler:
    - history: {'loss': [...], 'acc': [...], 'val_loss': [...], 'val_acc': [...]}
    - model_name: görselin dosya adı ve başlık etiketi olarak kullanılacak string
    """
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/saved/{model_name}_plot.png"

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close()

    print(f"📈 Training history saved to: {save_path}")