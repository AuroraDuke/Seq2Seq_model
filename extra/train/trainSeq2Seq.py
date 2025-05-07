import time
import os
import torch
import torch.nn as nn
from transformer_model.seq2seq import build_seq2seq_model
from util.metric import compute_metrics
from torch.optim import AdamW
from sklearn.metrics import f1_score
import torch.optim as optim

def train_seq2seq_model(attention_module,embedding_matrix,word2idx,
    output_dim,train_loader,val_loader,PAD_IDX,
    model_name='seq2seq_model',
    embedding_dim=100,hidden_dim=128,cell_type='lstm',EPOCHS=10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_model=True,
    num_layers=1,
    CLASS_W=[0.06, 0.89, 0.04]                    
):
    model = build_seq2seq_model(
        vocab_size=len(word2idx),
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        hidden_dim=64,  # This was your hidden_dim in the error trace
        attention_type='additive',
        pretrained_embedding=embedding_matrix,
        cell_type='lstm'
    )
    model=model.to(device)
    class_weights = torch.tensor(CLASS_W, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(class_weights)
    #optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.00001,
    alpha=0.9,
    eps=1e-8,
    weight_decay=1e-5,
    momentum=0.9,
    centered=False
    )

    history = {'loss': [], 'f1': [], 'val_loss': [], 'val_f1': []}
    start_train = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        all_preds, all_trues = [], []
    
        for src_batch, trg_batch in train_loader:
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            optimizer.zero_grad()
    
            output = model(src_batch)  # [batch_size, num_classes]
            loss = criterion(output, trg_batch)
            loss.backward()
            optimizer.step()
    
            # Metrikler için
            preds = output.argmax(dim=1).detach().cpu().numpy()
            trues = trg_batch.detach().cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(trues)
            epoch_loss += loss.item()
    
        train_f1 = f1_score(all_trues, all_preds, average='macro')
    
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_trues = [], []
    
        with torch.no_grad():
            for src_batch, trg_batch in val_loader:
                src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
                output = model(src_batch)
                loss = criterion(output, trg_batch)
    
                preds = output.argmax(dim=1).cpu().numpy()
                trues = trg_batch.cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(trues)
                val_loss += loss.item()
    
        val_f1 = f1_score(val_trues, val_preds, average='macro')
    
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
    
        [history[k].append(v) for k, v in zip(['loss', 'f1', 'val_loss', 'val_f1'],
                                             [avg_train_loss, train_f1, avg_val_loss, val_f1])]
    
        print(f"Epoch {epoch+1:02} | Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f} | Val_Loss: {avg_val_loss:.4f}, Val_F1: {val_f1:.4f}")
    end_train = time.time()
    train_time = (end_train - start_train) * 1000  # milliseconds

    # Save model
    if save_model:
        os.makedirs("models", exist_ok=True)
        save_path = f"models/{model_name}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\n✅ Model saved to: {save_path}")

    return model, history, train_time
