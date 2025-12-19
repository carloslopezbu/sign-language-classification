import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def imports():
    # Librerías necesarias
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import v2
    return Image, np, os, plt, torch, v2


@app.cell
def _():
    DINOV3_REPO_DIR = "./src/dinov3"  # Ruta al repo clonado
    WEIGHTS_PATH = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    WEIGHTS_PATH ="/Users/more/Desktop/tfg/models/dinov3/2dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"  # Tu archivo .pt


    IMAGE_PATH = "/Users/more/Desktop/tfg/datamining/samples/cat.webp"
    MODEL_NAME = "dinov3_vits16plus"  # Cambiar según tu modelo
    return DINOV3_REPO_DIR, IMAGE_PATH, MODEL_NAME, WEIGHTS_PATH


@app.cell
def _(DINOV3_REPO_DIR, MODEL_NAME, WEIGHTS_PATH, torch):
    model = torch.hub.load(
        repo_or_dir=DINOV3_REPO_DIR, 
        model=MODEL_NAME, 
        source="local", 
        weights=WEIGHTS_PATH, 
        verbose=True, 
        pretrained=True
    ).to("mps")
    return (model,)


@app.cell
def _(model):
    model.eval()
    return


@app.cell
def _(model):
    model.num_features
    return


@app.cell
def _(torch, v2):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize(mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225)),])
    return (transform,)


@app.cell
def _(IMAGE_PATH, Image, transform):
    img = Image.open(IMAGE_PATH).convert("RGB")
    img = transform(img).unsqueeze(0).to("mps")
    return (img,)


@app.cell
def _(img, model, torch):
    with torch.no_grad():
        res = model.get_intermediate_layers(img, n=range(12),norm=True)[-1]

    dim = res
    #res = res.view(dim, -1).permute(1, 0)
    return (res,)


@app.cell
def _(res):
    res.shape
    return


@app.cell
def _():
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from sklearn.svm import LinearSVC, SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import pandas as pd
    return (
        DataLoader,
        Dataset,
        SVC,
        TensorDataset,
        accuracy_score,
        classification_report,
        confusion_matrix,
        pd,
        sns,
    )


@app.cell
def _(Dataset, Image, os, pd):
    class PollenDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None, filter_classes=None, limit_per_class=None):
            self.img_dir = img_dir
            self.transform = transform

            # 1. Cargar CSV
            df = pd.read_csv(csv_file)

            # 2. Filtrar Clases (si se proporcionan)
            if filter_classes is not None:
                df = df[df['species'].isin(filter_classes)]

            # 3. Limitar muestras por clase (Balanced subsampling)
            if limit_per_class is not None:
                # Esto toma las primeras N filas de cada grupo (specie)
                df = df.groupby('species').head(limit_per_class).reset_index(drop=True)

            self.data = df

            # Definir clases y mapeo (Importante: ordenar para consistencia)
            self.classes = sorted(self.data['species'].unique().tolist())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            # Reporte
            print(f"📊 Dataset cargado ({os.path.basename(csv_file)}): {len(self.data)} imágenes, {len(self.classes)} clases.")
            print(f"   -> Distribución: {self.data['species'].value_counts().to_dict()}")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            img_name, label_name = row['sample'], row['species']
            img_path = os.path.join(self.img_dir, img_name)

            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new('RGB', (224, 224))

            if self.transform:
                image = self.transform(image)
            return image, self.class_to_idx[label_name]
    return (PollenDataset,)


@app.cell
def _(torch, v2):
    trans = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return


@app.cell
def _(DataLoader, PollenDataset, model, np, os, pd, torch, transform):
    def extract_features(dataloader, subset_name):
        print(f"🚀 Extrayendo features: {subset_name}...")
        features_list, labels_list = [], []

        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs = imgs.to(DEVICE)
                # DINOv3 forward -> suele devolver class token o features
                out = model(imgs)
                # Si devuelve tupla, cogemos el primero (CLS)
                if isinstance(out, (tuple, list)): out = out[0]
                # Si tiene dimensiones extra [B, N, D], aplanamos o cogemos CLS
                if out.dim() == 3: out = out[:, 0, :]

                features_list.append(out.cpu().numpy())
                labels_list.append(lbls.numpy())

        return np.concatenate(features_list), np.concatenate(labels_list)
    DEVICE="mps"
    BASE_DIR = "."
    BATCH_SIZE = 32
    TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
    VALID_CSV = os.path.join(BASE_DIR, "valid.csv")
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
    VALID_IMG_DIR = os.path.join(BASE_DIR, "valid")

    df_temp = pd.read_csv(TRAIN_CSV)
    N_CLASSES=5
    SAMPLES_PER_CLASS=200
    top_classes = df_temp['species'].value_counts().nlargest(N_CLASSES).index.tolist()
    print(f"\n🎯 Clases seleccionadas ({N_CLASSES}): {top_classes}")
    # Preparar Loaders
    ds_train = PollenDataset(
        TRAIN_CSV, TRAIN_IMG_DIR, transform, 
        filter_classes=top_classes, 
        limit_per_class=SAMPLES_PER_CLASS # 500 por clase
    )

    ds_val = PollenDataset(
        VALID_CSV, VALID_IMG_DIR, transform, 
        filter_classes=top_classes,
        limit_per_class=100 # Reducimos validación también para ir rápido (ej. 100 por clase)
    )

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    # Ejecutar extracción
    X_train, y_train = extract_features(dl_train, "TRAIN")
    X_val, y_val = extract_features(dl_val, "VALID")
    return DEVICE, X_train, X_val, ds_train, top_classes, y_train, y_val


@app.cell
def _(
    SVC,
    X_train,
    X_val,
    acc,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ds_train,
    plt,
    sns,
    y_train,
    y_val,
):
    print(f"✅ Extracción completada. Train shape: {X_train.shape}")

    # --- 4. ENTRENAR SVM ---
    print("🧠 Entrenando SVM Lineal...")
    clf_svm = SVC(degree=5,gamma=0.01, C=1000, max_iter=1000)
    clf_svm.fit(X_train, y_train)

    # Predicción
    y_pred_svm = clf_svm.predict(X_val)
    acc_svm = accuracy_score(y_val, y_pred_svm)

    # --- 5. RESULTADOS ---
    print("="*60)
    print(f"🏆 ACCURACY FINAL: {acc:.2%}")
    print("="*60)

    # Mostrar reporte
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred_svm, target_names=ds_train.classes))

    # Mostrar matriz de confusión
    cm_svm = confusion_matrix(y_val, y_pred_svm)
    fig_svm, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ds_train.classes, yticklabels=ds_train.classes)
    plt.title(f"Matriz de Confusión (Acc: {acc:.2%})")
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.show() # O mo.ui.p
    return


@app.cell
def _(
    DEVICE,
    DataLoader,
    TensorDataset,
    X_train,
    X_val,
    accuracy_score,
    classification_report,
    confusion_matrix,
    plt,
    sns,
    top_classes,
    torch,
    y_train,
    y_val,
):
    # Definir la MLP de 3 capas

    from torch import nn
    from torch import optim

    print(f"✅ Features extraídas. Shape: {X_train.shape}")

    # --- 4. DEFINIR Y ENTRENAR RED NEURONAL (MLP) ---

    # Crear DataLoaders para las Features (Entrenamiento mucho más rápido)
    train_feat_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_feat_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_feat_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_feat_ds, batch_size=64, shuffle=False)

    # Definir la MLP de 3 capas
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SimpleMLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),          # Evitar sobreajuste
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.layers(x)

    LR=0.001
    EPOCHS=100
    # Instanciar modelo
    input_dim = X_train.shape[1]
    mlp_model = SimpleMLP(input_dim, len(top_classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=LR)

    print(f"\n🧠 Entrenando MLP ({input_dim} -> 512 -> 256 -> {len(top_classes)})...")
    history = {'loss': [], 'acc': []}

    for epoch in range(EPOCHS):
        mlp_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = mlp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        history['loss'].append(running_loss/len(train_loader))
        history['acc'].append(epoch_acc)

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%")

    # --- 5. EVALUACIÓN Y RESULTADOS ---
    mlp_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = mlp_model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Métricas Finales
    acc = accuracy_score(all_labels, all_preds)

    print("="*60)
    print(f"🏆 ACCURACY FINAL (MLP): {acc:.2%}")
    print("="*60)

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=top_classes))

    # Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico de entrenamiento
    ax1.plot(history['loss'], label='Loss', color='red')
    ax1.set_title("Evolución del Loss")
    ax1.set_xlabel("Epochs")
    ax1.legend()

    # Matriz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=top_classes, yticklabels=top_classes, ax=ax2)
    ax2.set_title(f"Matriz de Confusión (Acc: {acc:.2%})")
    ax2.set_ylabel('Real')
    ax2.set_xlabel('Predicho')

    plt.tight_layout()
    plt.show()
    return (acc,)


@app.cell
def _(DataLoader, X_train, ds_train, plt, sns, top_classes, torch, y_train):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # --- 1. VISUALIZAR EMBEDDINGS ---
    def plot_embeddings(features, labels, class_names, title="Embeddings"):
        print(f"🎨 Generando t-SNE para {len(features)} puntos... (puede tardar un poco)")

        # PCA (Rápido, estructura global)
        pca = PCA(n_components=2)
        feat_pca = pca.fit_transform(features)

        # t-SNE (Lento, estructura local - ideal para clusters)
        # Perplexity bajo (5-30) para pocos datos
        tsne = TSNE(n_components=2, perplexity=min(30, len(features)//10), random_state=42, init='pca', learning_rate='auto')
        feat_tsne = tsne.fit_transform(features)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Colores
        palette = sns.color_palette("bright", len(class_names))

        # PCA Plot
        sns.scatterplot(x=feat_pca[:,0], y=feat_pca[:,1], hue=[class_names[i] for i in labels], 
                        palette=palette, ax=ax1, s=60, alpha=0.7)
        ax1.set_title("PCA (Proyección Lineal)")
        ax1.legend(title='Especie')

        # t-SNE Plot
        sns.scatterplot(x=feat_tsne[:,0], y=feat_tsne[:,1], hue=[class_names[i] for i in labels], 
                        palette=palette, ax=ax2, s=60, alpha=0.7)
        ax2.set_title("t-SNE (Proyección No Lineal)")
        ax2.legend(title='Especie')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    # Ejecutar visualización (usando tus variables anteriores)
    # Asegúrate de usar .cpu() si son tensores
    X_plot = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    y_plot = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train

    plot_embeddings(X_plot, y_plot, top_classes, title="Visualización de Features DINOv3")

    # --- 2. AUDITORÍA DE IMÁGENES (DEBUGGING) ---
    # Esto es CRÍTICO si los resultados son malos. 
    # Verifica si las imágenes se ven bien o son ruido/negro.

    def inspect_dataloader(dataloader, class_names):
        print("\n🕵️‍♂️ Inspeccionando qué entra a la red...")
        imgs, lbls = next(iter(dataloader))

        # Desnormalizar para visualizar (Mean/Std de ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        for i in range(8): # Mostrar 8 imágenes
            if i >= len(imgs): break
            img = imgs[i]

            # Deshacer normalización: img * std + mean
            img_vis = img * std + mean
            img_vis = torch.clamp(img_vis, 0, 1) # Asegurar rango [0,1]

            # Convertir a numpy [H, W, C]
            img_np = img_vis.permute(1, 2, 0).numpy()

            axes[i].imshow(img_np)
            axes[i].set_title(f"Label: {class_names[lbls[i]]}\nMin:{img.min():.1f} Max:{img.max():.1f}")
            axes[i].axis('off')

        plt.suptitle("Check de Datos (Batch de Entrenamiento)", fontsize=14)
        plt.tight_layout()
        plt.show()

    # Usar el dataloader que ya creaste
    inspect_dataloader(DataLoader(ds_train, batch_size=8, shuffle=True), top_classes)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
