import torch
from torch.utils.data import DataLoader
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from model import device

classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(12, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def evaluate(model, dataloader, criterion):
    metric=torchmetrics.MeanMetric().to(device)
    model.eval()
    metric.reset()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            metric.update(loss)
            predicted_classes = torch.argmax(y_pred, dim=1) 
            correct_predictions += (predicted_classes == y).sum().item()
            total_predictions += y.size(0)
    accuracy = correct_predictions / total_predictions
    return metric.compute().item(), accuracy


def train(model, optimizer, criterion, train_loader, valid_loader, n_epochs, scheduler=None) -> dict[str, list]:
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            train_loss = criterion(y_pred, y)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()
            predicted_classes = torch.argmax(y_pred, dim=1)
            correct_predictions += (predicted_classes == y).sum().item()
            total_predictions += y.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = correct_predictions / total_predictions
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        avg_valid_loss, avg_valid_acc = evaluate(model, valid_loader, criterion)
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_acc)

        if scheduler is not None:
            scheduler.step(avg_valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            break
        print(f"Epoch {epoch+1}/{n_epochs} | LR: {current_lr:.2e} | Train EntropyLoss: {avg_train_loss:.4e} (Acc: {avg_train_acc*100:.1f}%) | " 
            f"Valid EntropyLoss: {avg_valid_loss:.4e} (Acc: {avg_valid_acc*100:.1f}%)")
    return {
        "train_loss" : train_losses,
        "valid_loss" : valid_losses,
        "train_acc" : train_accuracies,
        "valid_acc" : valid_accuracies
    }


def plot_training(history, filename=None):
    x = 0
    # Zabezpieczenie na wypadek użycia Early Stopping (n_epochs może być krótsze niż planowaliśmy)
    actual_epochs = len(history["train_loss"]) - x
    epochs_train = np.arange(actual_epochs) + 0.5
    epochs_valid = np.arange(actual_epochs) + 1.0

    # Tworzymy płótno z dwoma wykresami obok siebie (1 wiersz, 2 kolumny)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- PANEL 1: Funkcja straty (CrossEntropy) ---
    ax1.plot(epochs_train, history["train_loss"], ".--", label="Training")
    ax1.plot(epochs_valid, history["valid_loss"], ".-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy")
    ax1.grid(True)
    ax1.set_title("Learning Curves (Loss)")
    ax1.legend()

    # --- PANEL 2: Celność (Accuracy) ---
    # Opcjonalnie mnożymy przez 100, jeśli chcesz widzieć procenty (zakładam, że acc jest od 0.0 do 1.0)
    ax2.plot(epochs_train, np.array(history["train_acc"][x:]) * 100, ".--", color="green", label="Training")
    ax2.plot(epochs_valid, np.array(history["valid_acc"][x:]) * 100, ".-", color="red", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy [%]")
    ax2.grid(True)
    ax2.set_title("Learning Curves (Accuracy)")
    ax2.legend()

    plt.tight_layout() # Zapobiega nakładaniu się napisów

    if filename is not None:
        plt.savefig(filename)
        
    plt.show()


def evaluate_model(model, dataloader, device, classes, dataset_name="Test", silent=False):
    print(f"\n--- Evaluating dataset: {dataset_name} ---")
    
    # 1. Przełączenie modelu w tryb "Tylko do odczytu"
    model.eval()
    
    all_preds = []
    all_true_labels = []
    
    # 2. Wyłączenie liczenia gradientów (oszczędza RAM i przyspiesza)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Predykcja (zwraca surowe logity)
            outputs = model(X)
            
            # Wybieramy indeks klasy z największym logitem
            # dim=1 oznacza szukanie maksimum wzdłuż osi klas
            _, preds = torch.max(outputs, dim=1)
            
            # Zrzucamy wyniki z powrotem na procesor (CPU) do listy
            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())
            
    # 3. Konwersja list do tablic NumPy dla scikit-learn
    all_preds = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)
    
    # 4. Obliczanie metryk
    acc = accuracy_score(all_true_labels, all_preds)
    if not silent:
        print(f"Accuracy: {acc:.2%}\n")
    
        print("Specified report:")
    # classification_report wylicza Precision, Recall i F1-Score dla każdej klasy osobno
    report = classification_report(
        all_true_labels, 
        all_preds, 
        target_names=classes, 
        zero_division=0 # Zapobiega błędom, gdy model w ogóle nie strzeli w jakąś klasę
    )
    if not silent:
        print(report)
    
    return all_true_labels, all_preds

def plot_cm(model, dataloader, classes):
    val_true, val_preds = evaluate_model(model, dataloader, device, classes, "Validation set", silent=True)

    cm = confusion_matrix(val_true, val_preds)


    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

    plt.title("Confusion Matrix - Validation set", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


def visualize_random_top2_predictions(model, dataloader, device, classes):
    print("Losowanie nowych zdjęć ze zbioru i generowanie predykcji...")
    model.eval()
    
    random_loader = DataLoader(dataloader.dataset, batch_size=32, shuffle=True)
    
    examples = {} 

    with torch.no_grad():
        # Iterujemy teraz po tymczasowym, potasowanym koszyku
        for X, y in random_loader:
            X_dev, y_dev = X.to(device), y.to(device)
            
            outputs = model(X_dev)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, k=2, dim=1)
            
            for i in range(len(y)):
                true_class_idx = y[i].item()
                
                if true_class_idx not in examples:
                    examples[true_class_idx] = {
                        'image': X[i].cpu(),
                        'true_label': classes[true_class_idx],
                        'top2_p': top_probs[i].cpu().numpy(),
                        'top2_idx': top_indices[i].cpu().numpy()
                    }
                
                if len(examples) == 10:
                    break
            if len(examples) == 10:
                break

    # Rysowanie wykresu
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten() 
    
    for idx in range(10): 
        if idx in examples:
            ex = examples[idx]
            
            img = ex['image'] / 2 + 0.5 
            npimg = img.numpy()
            npimg = np.transpose(npimg, (1, 2, 0))
            
            ax = axes[idx]
            ax.imshow(npimg)
            ax.axis('off')
            
            true_name = ex['true_label']
            p1_name = classes[ex['top2_idx'][0]]
            p1_prob = ex['top2_p'][0]
            p2_name = classes[ex['top2_idx'][1]]
            p2_prob = ex['top2_p'][1]
            
            title_color = 'darkgreen' if p1_name == true_name else 'darkred'
            title = f"1. {p1_name} ({p1_prob:.1%})\n"
            title += f"2. {p2_name} ({p2_prob:.1%})"
            
            ax.set_title(title, color=title_color, fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    plt.show()