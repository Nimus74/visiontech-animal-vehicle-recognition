"""
Image Recognition - Animal vs Vehicle Recognition
Interfaccia Streamlit per classificazione immagini e visualizzazione risultati.

Esegui con: streamlit run app.py
"""

import os
import sys
import random
from typing import Optional, Tuple

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Aggiungi il path del progetto a sys.path per permettere gli import
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.streamlit_utils import (
    load_model,
    predict_image_streamlit,
    load_metrics,
    load_training_history,
    get_error_summary,
    get_misclassified_images,
)
from src.utils import load_config, pick_device
from src.dataset import CIFAR10AnimalVehicle
from torchvision import transforms

# Configurazione pagina
st.set_page_config(
    page_title="VisionTech - Animal vs Vehicle Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato per migliorare l'aspetto
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached():
    """
    Carica il modello con caching Streamlit.
    Il modello viene caricato una sola volta e riutilizzato.
    """
    config = load_config("configs/default.yaml")
    ckpt_path = config["outputs"]["checkpoint_path"]
    
    if not os.path.exists(ckpt_path):
        return None, None, "Checkpoint non trovato. Esegui prima il training con: python src/train.py"
    
    try:
        device = pick_device(config["train"]["device"])
        model, info = load_model(
            ckpt_path,
            device,
            dropout=float(config["model"]["dropout"])
        )
        return model, device, None
    except Exception as e:
        return None, None, f"Errore nel caricamento del modello: {str(e)}"


def render_classification_page(model: torch.nn.Module, device: torch.device):
    """Pagina principale per la classificazione di immagini."""
    st.title("🔍 Classificazione Immagini")
    st.markdown("Carica un'immagine per classificarla come **Animal** o **Vehicle**")
    
    # Tabs per diverse modalità di input
    tab1, tab2, tab3 = st.tabs(["📤 Upload Immagine", "📷 Webcam", "🎲 Esempi Test Set"])
    
    with tab1:
        st.subheader("Carica un'immagine")
        uploaded_file = st.file_uploader(
            "Scegli un'immagine...",
            type=['png', 'jpg', 'jpeg'],
            help="Formati supportati: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Mostra l'immagine caricata
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Immagine caricata", use_container_width=True)
            
            with col2:
                if st.button("🔮 Classifica", type="primary", use_container_width=True):
                    with st.spinner("Classificazione in corso..."):
                        label, prob_animal, prob_vehicle = predict_image_streamlit(
                            model, device, image
                        )
                    
                    # Mostra risultato
                    st.markdown("### Risultato")
                    
                    # Colore in base alla confidence
                    confidence = max(prob_animal, prob_vehicle)
                    if confidence > 0.8:
                        st.success(f"**Predizione: {label}** (Confidence: {confidence*100:.1f}%)")
                    elif confidence > 0.6:
                        st.warning(f"**Predizione: {label}** (Confidence: {confidence*100:.1f}%)")
                    else:
                        st.error(f"**Predizione: {label}** (Confidence: {confidence*100:.1f}%) - Bassa confidence!")
                    
                    # Barre di probabilità
                    st.markdown("#### Probabilità")
                    st.progress(prob_animal, text=f"Animal: {prob_animal*100:.1f}%")
                    st.progress(prob_vehicle, text=f"Vehicle: {prob_vehicle*100:.1f}%")
                    
                    # Grafico a barre
                    prob_data = pd.DataFrame({
                        'Classe': ['Animal', 'Vehicle'],
                        'Probabilità': [prob_animal, prob_vehicle]
                    })
                    st.bar_chart(prob_data.set_index('Classe'))
    
    with tab2:
        st.subheader("Webcam")
        camera_image = st.camera_input("Scatta una foto con la webcam")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption="Foto scattata", use_container_width=True)
            
            if st.button("🔮 Classifica dalla Webcam", type="primary"):
                with st.spinner("Classificazione in corso..."):
                    label, prob_animal, prob_vehicle = predict_image_streamlit(
                        model, device, image
                    )
                
                confidence = max(prob_animal, prob_vehicle)
                if confidence > 0.8:
                    st.success(f"**Predizione: {label}** (Confidence: {confidence*100:.1f}%)")
                else:
                    st.warning(f"**Predizione: {label}** (Confidence: {confidence*100:.1f}%)")
                
                st.progress(prob_animal, text=f"Animal: {prob_animal*100:.1f}%")
                st.progress(prob_vehicle, text=f"Vehicle: {prob_vehicle*100:.1f}%")
    
    with tab3:
        st.subheader("Esempi dal Test Set")
        st.markdown("Classifica immagini casuali dal dataset CIFAR-10")
        
        # Inizializza session state se non esiste
        if 'test_image' not in st.session_state:
            st.session_state.test_image = None
            st.session_state.test_idx = None
            st.session_state.test_true_label = None
        
        # Bottone per caricare nuova immagine casuale
        if st.button("🎲 Carica Immagine Casuale"):
            try:
                # Carica il test set
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                test_dataset = CIFAR10AnimalVehicle(
                    root='data', 
                    train=False, 
                    transform=test_transform, 
                    download=False
                )
                
                # Scegli un indice casuale
                idx = random.randint(0, len(test_dataset) - 1)
                img_tensor, true_label, cifar_label = test_dataset[idx]
                
                # Denormalizza per visualizzazione
                denormalize = transforms.Normalize(
                    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
                    std=[1/0.2023, 1/0.1994, 1/0.2010]
                )
                img_display = denormalize(img_tensor)
                img_display = torch.clamp(img_display, 0, 1)
                img_pil = transforms.ToPILImage()(img_display)
                
                # Salva nello session state
                st.session_state.test_image = img_pil
                st.session_state.test_idx = idx
                st.session_state.test_true_label = true_label
                
            except Exception as e:
                st.error(f"Errore nel caricamento del test set: {str(e)}")
                st.info("Assicurati che il dataset CIFAR-10 sia stato scaricato durante il training.")
        
        # Mostra l'immagine se presente nello session state
        if st.session_state.test_image is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Riduci la dimensione dell'immagine
                st.image(
                    st.session_state.test_image, 
                    caption=f"Immagine #{st.session_state.test_idx} dal test set",
                    width=200  # Dimensione fissa più piccola
                )
                true_label_name = "Animal" if st.session_state.test_true_label == 0 else "Vehicle"
                st.info(f"**Label vera:** {true_label_name}")
            
            with col2:
                if st.button("🔮 Classifica", type="primary", use_container_width=True):
                    with st.spinner("Classificazione in corso..."):
                        label, prob_animal, prob_vehicle = predict_image_streamlit(
                            model, device, st.session_state.test_image
                        )
                    
                    confidence = max(prob_animal, prob_vehicle)
                    is_correct = (label == true_label_name)
                    
                    if is_correct:
                        st.success(f"✅ **Corretto!** Predizione: {label} (Confidence: {confidence*100:.1f}%)")
                    else:
                        st.error(f"❌ **Errato!** Predetto: {label}, Vero: {true_label_name}")
                    
                    st.progress(prob_animal, text=f"Animal: {prob_animal*100:.1f}%")
                    st.progress(prob_vehicle, text=f"Vehicle: {prob_vehicle*100:.1f}%")
                    
                    # Grafico a barre delle probabilità
                    prob_data = pd.DataFrame({
                        'Classe': ['Animal', 'Vehicle'],
                        'Probabilità': [prob_animal, prob_vehicle]
                    })
                    st.bar_chart(prob_data.set_index('Classe'))
        else:
            st.info("👆 Clicca su 'Carica Immagine Casuale' per iniziare")


def render_metrics_page():
    """Pagina dashboard con metriche e grafici."""
    st.title("📊 Dashboard Metriche")
    
    config = load_config("configs/default.yaml")
    metrics_path = os.path.join(config["outputs"]["metrics_dir"], "evaluation.json")
    history_path = os.path.join(config["outputs"]["metrics_dir"], "training_history.json")
    plots_dir = config["outputs"]["plots_dir"]
    ckpt_path = config["outputs"]["checkpoint_path"]
    
    # Carica metriche
    metrics = load_metrics(metrics_path)
    history = load_training_history(history_path)
    
    if metrics is None:
        st.warning("⚠️ Metriche non trovate. Esegui prima: python src/evaluate.py")
        return
    
    # Metric cards principali
    st.subheader("Metriche Principali")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision (Vehicle)", f"{metrics['precision_vehicle']:.2%}")
    with col3:
        st.metric("Recall (Vehicle)", f"{metrics['recall_vehicle']:.2%}")
    with col4:
        st.metric("F1-Score (Vehicle)", f"{metrics['f1_vehicle']:.2%}")
    
    # Informazioni checkpoint
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 'unknown')
            best_acc = checkpoint.get('best_val_acc', checkpoint.get('best_acc', 'unknown'))
            
            with st.expander("ℹ️ Informazioni Modello"):
                st.write(f"**Epoca migliore:** {epoch}")
                st.write(f"**Best Validation Accuracy:** {best_acc:.4f}" if isinstance(best_acc, float) else f"**Best Validation Accuracy:** {best_acc}")
        except:
            pass
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
    else:
        # Mostra confusion matrix da dati
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Animal', 'Vehicle'],
                yticklabels=['Animal', 'Vehicle'],
                ax=ax
            )
            ax.set_xlabel('Predizione')
            ax.set_ylabel('Verità')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    
    # Training curves
    if history:
        st.subheader("Training Curves")
        
        tab1, tab2 = st.tabs(["Loss", "Metriche"])
        
        with tab1:
            if "train_loss" in history and "val_loss" in history:
                epochs = range(1, len(history["train_loss"]) + 1)
                loss_data = pd.DataFrame({
                    'Epoca': epochs,
                    'Train Loss': history["train_loss"],
                    'Validation Loss': history["val_loss"]
                })
                st.line_chart(loss_data.set_index('Epoca'))
        
        with tab2:
            if "val_acc" in history:
                epochs = range(1, len(history["val_acc"]) + 1)
                metrics_data = pd.DataFrame({
                    'Epoca': epochs,
                    'Accuracy': history["val_acc"],
                    'Precision': history.get("val_precision", []),
                    'Recall': history.get("val_recall", []),
                    'F1-Score': history.get("val_f1", [])
                })
                if len(metrics_data) > 0:
                    st.line_chart(metrics_data.set_index('Epoca'))
        
        # Grafici salvati
        st.subheader("Grafici Salvati")
        col1, col2 = st.columns(2)
        
        with col1:
            curves_path = os.path.join(plots_dir, "training_curves.png")
            if os.path.exists(curves_path):
                st.image(curves_path, caption="Training Curves", use_container_width=True)
        
        with col2:
            comparison_path = os.path.join(plots_dir, "metrics_comparison.png")
            if os.path.exists(comparison_path):
                st.image(comparison_path, caption="Confronto Metriche", use_container_width=True)
    
    # Classification Report
    if "classification_report" in metrics:
        with st.expander("📋 Classification Report Completo"):
            st.text(metrics["classification_report"])


def render_errors_page():
    """Pagina analisi degli errori."""
    st.title("🔍 Analisi Errori")
    
    config = load_config("configs/default.yaml")
    misclassified_dir = config["outputs"]["misclassified_dir"]
    summary_path = os.path.join(misclassified_dir, "summary.json")
    
    summary = get_error_summary(summary_path)
    
    if summary is None:
        st.warning("⚠️ Analisi errori non disponibile. Esegui prima: python src/error_analysis.py")
        return
    
    # Statistiche generali
    st.subheader("Statistiche Generali")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Totale Errori", summary.get("total_errors", "N/A"))
    with col2:
        error_rate = summary.get("error_rate", 0)
        st.metric("Tasso di Errore", f"{error_rate:.2%}" if isinstance(error_rate, float) else "N/A")
    with col3:
        st.metric("Immagini Salvate", summary.get("saved_misclassified", 0))
    
    # Top pattern di confusione
    st.subheader("Top Pattern di Confusione")
    if "top_confusions" in summary and len(summary["top_confusions"]) > 0:
        confusions_df = pd.DataFrame(summary["top_confusions"][:10])
        st.dataframe(
            confusions_df[["cifar_name", "true_label", "pred_label", "count"]],
            column_config={
                "cifar_name": "Classe CIFAR",
                "true_label": "Label Vera",
                "pred_label": "Label Predetta",
                "count": "Conteggio"
            },
            use_container_width=True,
            hide_index=True
        )
    
    # Griglia immagini errate
    st.subheader("Immagini Classificate Erroneamente")
    
    error_images = get_misclassified_images(misclassified_dir, max_images=30)
    
    if len(error_images) == 0:
        st.info("Nessuna immagine trovata nella directory misclassified.")
    else:
        # Filtro per tipo di errore
        filter_type = st.selectbox(
            "Filtra per tipo di errore",
            ["Tutti", "Animal → Vehicle", "Vehicle → Animal"]
        )
        
        # Filtra immagini
        filtered_images = []
        for img_path in error_images:
            filename = os.path.basename(img_path)
            if filter_type == "Tutti":
                filtered_images.append(img_path)
            elif filter_type == "Animal → Vehicle" and "_true0_pred1_" in filename:
                filtered_images.append(img_path)
            elif filter_type == "Vehicle → Animal" and "_true1_pred0_" in filename:
                filtered_images.append(img_path)
        
        # Mostra griglia
        num_cols = 4
        num_rows = (len(filtered_images) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                img_idx = row * num_cols + col_idx
                if img_idx < len(filtered_images):
                    with cols[col_idx]:
                        try:
                            img = Image.open(filtered_images[img_idx])
                            filename = os.path.basename(filtered_images[img_idx])
                            
                            # Estrai informazioni dal filename
                            parts = filename.replace('.png', '').split('_')
                            true_label = "Animal" if "true0" in filename else "Vehicle"
                            pred_label = "Animal" if "pred0" in filename else "Vehicle"
                            
                            st.image(img, use_container_width=True)
                            st.caption(f"Vero: {true_label} → Predetto: {pred_label}")
                        except Exception as e:
                            st.error(f"Errore nel caricamento: {str(e)}")


def render_info_page():
    """Pagina informazioni sul progetto."""
    st.title("ℹ️ Informazioni Progetto")
    
    st.markdown("""
    ## Image Recognition - Animal vs Vehicle Recognition
    
    Sistema di riconoscimento immagini basato su **Convolutional Neural Network (CNN)** 
    per classificare immagini in **Animale** vs **Veicolo**.
    
    ### 🎯 Obiettivo
    
    Il progetto mira a sviluppare un sistema automatico di riconoscimento che possa:
    - Classificare immagini (da telecamere stradali / stream video) in Animal o Vehicle
    - Ridurre incidenti dovuti ad attraversamento improvviso di animali
    - Supportare analisi statistiche e politiche di sicurezza stradale
    
    ### 📊 Dataset
    
    Utilizziamo il dataset **CIFAR-10** convertito in classificazione binaria:
    
    **Vehicle (classe 1)**: airplane, automobile, ship, truck
    
    **Animal (classe 0)**: bird, cat, deer, dog, frog, horse
    
    ### 🏗️ Architettura del Modello
    
    Il modello implementa una **CNN custom** con PyTorch:
    
    ```
    SimpleCNN:
    ├── Feature Extractor (3 blocchi convoluzionali)
    │   ├── Conv2D(3→32) + BatchNorm + ReLU + MaxPool
    │   ├── Conv2D(32→64) + BatchNorm + ReLU + MaxPool
    │   └── Conv2D(64→128) + BatchNorm + ReLU + MaxPool
    └── Classifier (MLP)
        ├── Flatten
        ├── Linear(2048→256) + ReLU + Dropout(0.25)
        └── Linear(256→2) → [Animal, Vehicle]
    ```
    
    ### 📈 Metriche
    
    - **Accuracy**: Percentuale di immagini classificate correttamente
    - **Precision**: Qualità delle predizioni positive (Vehicle)
    - **Recall**: Capacità di trovare tutti i Vehicle
    - **F1-Score**: Media armonica di Precision e Recall
    
    ### 🚀 Come Usare
    
    1. **Training**: `python src/train.py`
    2. **Valutazione**: `python src/evaluate.py`
    3. **Visualizzazione**: `python src/visualize.py`
    4. **Analisi Errori**: `python src/error_analysis.py`
    5. **Interfaccia Web**: `streamlit run app.py`
    
    ### 📝 Note Tecniche
    
    - Il modello è addestrato su immagini 32x32 pixel RGB
    - Supporta CPU, CUDA e MPS (Apple Silicon)
    - Il training usa split train/val/test corretto (best practice ML)
    - Early stopping previene overfitting
    
    ---
    
    **Sviluppato da Francesco Scarano - Progetto Master AI Engineering**
    """)


def main():
    """Funzione principale dell'app Streamlit."""
    # Header
    st.markdown('<div class="main-header">Image Recognition</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Animal vs Vehicle Recognition</p>', unsafe_allow_html=True)
    
    # Carica modello
    model, device, error = load_model_cached()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigazione")
    pages = {
        "🔍 Classificazione": render_classification_page,
        "📊 Metriche": render_metrics_page,
        "🔍 Analisi Errori": render_errors_page,
        "ℹ️ Informazioni": render_info_page,
    }
    
    selected_page = st.sidebar.selectbox("Scegli una pagina", list(pages.keys()))
    
    # Mostra errore se il modello non è disponibile (solo per pagina classificazione)
    if selected_page == "🔍 Classificazione":
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Esegui prima il training con: `python src/train.py`")
            return
        if model is None:
            st.error("❌ Errore nel caricamento del modello")
            return
    
    # Renderizza la pagina selezionata
    if selected_page == "🔍 Classificazione":
        pages[selected_page](model, device)
    else:
        pages[selected_page]()


if __name__ == "__main__":
    main()
