# Image Recognition — Animal vs Vehicle Recognition

Sistema di riconoscimento immagini basato su **Convolutional Neural Network (CNN)** per classificare immagini in **Animale** vs **Veicolo**, con l'obiettivo di supportare sistemi di monitoraggio urbano e prevenzione incidenti (es. segnaletica elettronica di avviso).

## 📋 Indice

- [Obiettivo del Progetto](#obiettivo-del-progetto)
- [Dataset](#dataset)
- [Architettura del Modello](#architettura-del-modello)
- [Installazione](#installazione)
- [Struttura del Progetto](#struttura-del-progetto)
- [Utilizzo](#utilizzo)
- [Metriche e Valutazione](#metriche-e-valutazione)
- [Risultati](#risultati)

## 🎯 Obiettivo del Progetto

Il progetto mira a sviluppare un sistema automatico di riconoscimento che possa:

- **Classificare immagini** (da telecamere stradali / stream video) in:
  - **Animal (0)**: animali che potrebbero attraversare la strada
  - **Vehicle (1)**: veicoli in transito
- **Ridurre incidenti** dovuti ad attraversamento improvviso di animali
- **Supportare analisi statistiche** e politiche di sicurezza stradale
- **Fornire un sistema** scalabile per monitoraggio urbano in tempo reale

### Benefici

- ✅ **Automazione dei Processi**: Rilevamento automatico e in tempo reale
- ✅ **Alta Precisione**: Utilizzo di CNN per garantire accuratezza nella classificazione
- ✅ **Efficienza Operativa**: Processamento rapido di grandi volumi di dati
- ✅ **Maggiore Sicurezza**: Prevenzione incidenti e protezione di animali e veicoli

## 📊 Dataset

Utilizziamo il dataset **CIFAR-10** che contiene:
- **50,000 immagini di training** (32x32 pixel, RGB)
- **10,000 immagini di test** (32x32 pixel, RGB)
- **10 classi originali** che vengono convertite in un problema binario

### Conversione in Classificazione Binaria

**Vehicle (classe 1)**:
- `airplane` (aereo)
- `automobile` (automobile)
- `ship` (nave)
- `truck` (camion)

**Animal (classe 0)**:
- `bird` (uccello)
- `cat` (gatto)
- `deer` (cervo)
- `dog` (cane)
- `frog` (rana)
- `horse` (cavallo)

### Split dei Dati

Il progetto segue le **best practices del Machine Learning**:

- **Training Set**: 80% del dataset originale (40,000 immagini)
- **Validation Set**: 20% del training set (10,000 immagini) - usato durante il training
- **Test Set**: 10,000 immagini - usato SOLO per la valutazione finale

⚠️ **Importante**: Il test set non viene mai usato durante il training per evitare overfitting!

## 🏗️ Architettura del Modello

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

**Caratteristiche**:
- **3 blocchi convoluzionali** per estrarre features
- **Batch Normalization** per stabilizzare il training
- **Dropout** (25%) per prevenire overfitting
- **Output**: 2 logits (probabilità per Animal e Vehicle)

## 🚀 Installazione

### Opzione 1: Google Colab

Il modo più semplice per iniziare è usare **Google Colab**, che offre GPU gratuite!

1. Apri il notebook `notebooks/image_recognition.ipynb` su Google Colab
2. Segui le istruzioni nel notebook
3. Tutto è già configurato e pronto all'uso!

**Vantaggi di Colab**:
- ✅ Nessuna installazione locale necessaria
- ✅ GPU gratuita per training veloce
- ✅ Tutto funziona nel browser
- ✅ Notebook interattivo con spiegazioni

### Opzione 2: Installazione Locale

### Prerequisiti

- Python 3.8 o superiore
- pip (gestore pacchetti Python)

### Passi per l'Installazione

1. **Clona o scarica il repository**

```bash
cd visiontech-animal-vehicle-recognition
```

2. **Crea un ambiente virtuale** (consigliato)

```bash
# Su macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Su Windows
python -m venv .venv
.venv\Scripts\activate
```

3. **Installa le dipendenze**

```bash
pip install -r requirements.txt
```

Le dipendenze principali includono:
- `torch` e `torchvision`: Framework per deep learning
- `numpy`: Calcoli numerici
- `matplotlib` e `seaborn`: Visualizzazioni
- `scikit-learn`: Metriche di valutazione
- `opencv-python`: Elaborazione immagini
- `tqdm`: Barre di progresso
- `pyyaml`: Lettura configurazioni
- `streamlit`: Interfaccia web interattiva (opzionale)
- `pandas`: Manipolazione dati per visualizzazioni

## 📁 Struttura del Progetto

```
visiontech-animal-vehicle-recognition/
├── configs/
│   └── default.yaml          # Configurazione parametri (learning rate, batch size, etc.)
├── data/                      # Dataset CIFAR-10 (scaricato automaticamente)
├── models/
│   ├── __init__.py
│   └── cnn.py                # Architettura CNN
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Caricamento e preprocessing dati
│   ├── train.py              # Script di training
│   ├── evaluate.py           # Valutazione finale sul test set
│   ├── error_analysis.py    # Analisi degli errori
│   ├── infer.py              # Inference su nuove immagini
│   ├── visualize.py          # Generazione grafici
│   ├── utils.py              # Funzioni di utilità
│   └── streamlit_utils.py    # Funzioni helper per Streamlit
├── app.py                     # Interfaccia web Streamlit (NUOVO)
├── notebooks/
│   └── image_recognition.ipynb  # Notebook Jupyter per Google Colab
├── outputs/                   # Risultati generati (creato automaticamente)
│   ├── checkpoints/          # Modelli salvati
│   ├── metrics/              # Metriche JSON
│   ├── plots/                # Grafici
│   └── misclassified/        # Immagini classificate erroneamente
├── requirements.txt          # Dipendenze Python
└── README.md                 # Questo file
```

## 💻 Utilizzo

### 1. Training del Modello

Per addestrare il modello da zero:

```bash
# Assicurati di essere nella root del progetto
cd visiontech-animal-vehicle-recognition

# Esegui il training
python src/train.py
```

**Nota**: Gli script sono configurati per funzionare da qualsiasi directory, ma è consigliabile eseguirli dalla root del progetto.

Il training:
- Scarica automaticamente CIFAR-10 se non presente
- Divide i dati in train/val/test
- Addestra per 20 epoche (configurabile in `configs/default.yaml`)
- Salva il miglior modello basato su validation accuracy
- Supporta early stopping per evitare overfitting

**Output**:
- Modello salvato in `outputs/checkpoints/best.pt`
- Storia delle metriche in `outputs/metrics/training_history.json`

### 2. Valutazione del Modello

Per valutare il modello sul test set:

```bash
python src/evaluate.py
```

Calcola e stampa:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

**Output**: `outputs/metrics/evaluation.json`

### 3. Visualizzazione dei Risultati

Per generare i grafici delle metriche:

```bash
python src/visualize.py
```

Genera:
- Training curves (loss e accuracy)
- Confronto metriche di validazione
- Confusion matrix visualizzata

**Output**: Grafici salvati in `outputs/plots/`

### 4. Analisi degli Errori

Per analizzare le immagini classificate erroneamente:

```bash
python src/error_analysis.py
```

Salva:
- Immagini errate denormalizzate (visualizzabili)
- Griglia visuale degli errori
- Statistiche sui pattern di confusione

**Output**: `outputs/misclassified/`

### 5. Inference su Nuove Immagini

Per classificare una singola immagine:

```bash
python src/infer.py --image path/to/image.jpg
```

Per usare la webcam (demo):

```bash
python src/infer.py --webcam
```

## 📈 Metriche e Valutazione

### Metriche Utilizzate

1. **Accuracy**: Percentuale di immagini classificate correttamente
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: Qualità delle predizioni positive (Vehicle)
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall**: Capacità di trovare tutti i Vehicle
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Media armonica di Precision e Recall
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```

5. **Confusion Matrix**: Matrice che mostra:
   - **True Positives (TP)**: Vehicle correttamente identificati
   - **True Negatives (TN)**: Animal correttamente identificati
   - **False Positives (FP)**: Animal scambiati per Vehicle
   - **False Negatives (FN)**: Vehicle scambiati per Animal

### Interpretazione delle Metriche

- **Alta Precision**: Pochi falsi allarmi (Animal scambiati per Vehicle)
- **Alto Recall**: Rileva la maggior parte dei Vehicle (importante per sicurezza)
- **Alto F1**: Bilanciamento tra Precision e Recall
- **Alta Accuracy**: Buona performance generale

## 📊 Risultati

### Configurazione Standard

- **Epoche**: 20
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Dropout**: 0.25
- **Optimizer**: AdamW

### Risultati Attesi

Con la configurazione standard, ci si aspetta:

- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%
- **Precision (Vehicle)**: ~85-90%
- **Recall (Vehicle)**: ~85-90%
- **F1-Score**: ~85-90%

⚠️ **Nota**: I risultati possono variare leggermente a causa della randomicità nell'inizializzazione dei pesi e nello split dei dati.

### Analisi degli Errori

Gli errori più comuni includono:
- Confusione tra animali e veicoli con pattern di sfondo simili
- Immagini con bassa risoluzione o scarsa illuminazione
- Oggetti parzialmente visibili o occlusi

## 🔧 Configurazione Avanzata

Modifica `configs/default.yaml` per personalizzare:

```yaml
data:
  batch_size: 128          # Dimensione del batch
  val_split_ratio: 0.2     # Percentuale validation set

train:
  epochs: 20               # Numero di epoche
  lr: 0.001                # Learning rate
  early_stopping_patience: 5  # Epoche senza miglioramento prima di fermarsi

model:
  dropout: 0.25            # Tasso di dropout
```

## 📝 Note Tecniche

### Best Practices Implementate

1. ✅ **Train/Val/Test Split Corretto**: Il test set non viene mai usato durante il training
2. ✅ **Early Stopping**: Previene overfitting fermando il training quando necessario
3. ✅ **Data Augmentation**: Random crop e flip per aumentare la varietà dei dati
4. ✅ **Batch Normalization**: Stabilizza il training e accelera la convergenza
5. ✅ **Dropout**: Riduce overfitting nel classifier
6. ✅ **Metriche Complete**: Monitoraggio di accuracy, precision, recall, F1

### Limitazioni

- **Risoluzione**: CIFAR-10 ha immagini 32x32, quindi dettagli fini possono essere persi
- **Dataset**: CIFAR-10 è un dataset sintetico, risultati reali possono variare
- **Classi**: Solo 2 classi (Animal/Vehicle), non distingue tipi specifici

## 6. Interfaccia Web Streamlit (Estensione opzionale)

La consegna del progetto prevede lo sviluppo del modello di classificazione, la valutazione delle prestazioni e l’analisi dei risultati.  
In aggiunta, è stata sviluppata un’interfaccia web basata su **Streamlit** come **estensione opzionale**, con l’obiettivo di migliorare la fruibilità del sistema e consentire una dimostrazione interattiva delle funzionalità implementate.

L’interfaccia consente di utilizzare il modello addestrato senza ricorrere a notebook o riga di comando, risultando particolarmente utile in fase di presentazione, validazione e analisi esplorativa dei risultati.

#### Avvio dell’interfaccia

```bash
# Assicurati di aver installato streamlit
pip install streamlit

# Avvia l'applicazione web
streamlit run app.py
```

# L’applicazione sarà disponibile all’indirizzo:
http://localhost:8501

**Funzionalità principali**
 - Classificazione delle immagini
 - Inferenza su immagini caricate dall’utente, immagini di esempio o stream da webcam, con visualizzazione immediata della predizione (Animale / Veicolo).
 - Dashboard delle metriche
 - Visualizzazione delle principali metriche di valutazione del modello:
    - Accuracy
    - Precision
	- Recall
	- F1-score
	- Confusion Matrix
 - Inclusa la visualizzazione delle curve di training (loss e accuracy).
 - Analisi degli errori
 - Esplorazione delle immagini classificate erroneamente, con identificazione dei pattern di confusione più frequenti tra le categorie.
 - Sezione informativa
 - Descrizione del progetto, dell’architettura del modello e delle principali scelte progettuali.

**Caratteristiche dell’interfaccia** 
 - Interfaccia intuitiva orientata all’analisi dei dati
 - Visualizzazione interattiva dei risultati e delle metriche
 - Supporto per inferenza su immagini singole e flussi video
 - Layout responsive, adatto a demo e presentazioni


## 📄 Licenza

Questo progetto è stato sviluppato per scopi educativi e di ricerca.

## 👨‍💻 Autore

Francesco Scarano
