import time
import os
import torch
import torch.nn as nn
from transformer_model.seq2seq import Encoder, Decoder, Seq2Seq
from util.metric import compute_metrics

def train_seq2seq_model(
    attention_module,
    embedding_matrix,
    word2idx,
    output_dim,
    train_loader,
    val_loader,
    PAD_IDX,
    model_name='seq2seq_model',
    embedding_dim=100,
    hidden_dim=128,
    cell_type='lstm',
    EPOCHS=10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_model=True
):
    # Encoder
    encoder = Encoder(
        input_dim=len(word2idx),
        emb_dim=embedding_dim,
        hidden_dim=hidden_dim,
        cell_type=cell_type,
        pretrained_embedding=embedding_matrix,
        attention=attention_module
    )

    # Decoder
    decoder = Decoder(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        cell_type=cell_type,
        attention=attention_module
    )

    # Seq2Seq model
    model = Seq2Seq(encoder, decoder).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    start_train = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        for src_batch, trg_batch in train_loader:
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            optimizer.zero_grad()
            output = model(src_batch)
            loss = criterion(output, trg_batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = output.argmax(dim=1)
                metrics = compute_metrics(trg_batch, preds)

            epoch_loss += loss.item()
            epoch_acc += metrics['accuracy']

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for src_batch, trg_batch in val_loader:
                src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
                output = model(src_batch)
                loss = criterion(output, trg_batch)
                preds = output.argmax(dim=1)
                metrics = compute_metrics(trg_batch, preds)
                val_loss += loss.item()
                val_acc += metrics['accuracy']

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        [history[k].append(v) for k, v in zip(['loss', 'acc', 'val_loss', 'val_acc'], [avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])]
        print(f"Epoch {epoch+1:02} - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}, Val_Loss: {avg_val_loss:.4f}, Val_Acc: {avg_val_acc:.4f}")

    end_train = time.time()
    train_time = (end_train - start_train) * 1000  # milliseconds

    # Save model
    if save_model:
        os.makedirs("models", exist_ok=True)
        save_path = f"models/{model_name}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\n✅ Model saved to: {save_path}")

    return model, history, train_time
