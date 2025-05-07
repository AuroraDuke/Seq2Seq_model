import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from util.metric import compute_metrics
def evaluate_and_plot_confusion_matrix(model, src_tensor, trg_tensor, device, model_name='model', show_plot=True):
    """
    Modeli değerlendirir, metrikleri hesaplar ve Confusion Matrix'i kaydeder.

    Parametreler:
    - model: eğitilmiş PyTorch modeli
    - src_tensor: giriş tensor (X)
    - trg_tensor: hedef tensor (y)
    - compute_metrics_fn: metrik hesaplayan fonksiyon #compute_metrics_fn
    - device: 'cuda' veya 'cpu'
    - model_name: başlıkta ve dosyada kullanılacak model adı
    - show_plot: True ise plt.show() ile ekrana da çizer

    Dönüş:
    - metrics: {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}
    """

    model.eval()
    with torch.no_grad():
        src = src_tensor.to(device)
        trg = trg_tensor.to(device)

        output = model(src)
        y_pred = output.argmax(dim=1)
        y_true = trg

    # Metrikleri hesapla
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n🔍 Test Metrikleri - {model_name}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")

    # 📁 Kaydetme klasörünü oluştur
    os.makedirs("plots/saved", exist_ok=True)
    save_path = f"plots/saved/{model_name}_csmtx.png"
    plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"📌 Confusion matrix saved to: {save_path}")
    return metrics
