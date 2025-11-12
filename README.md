# BiLSTM for Unbiased Toxic Comment Classification

A Bidirectional LSTM model designed to classify toxic comments without bias towards minority groups. This implementation uses pre-trained GloVe embeddings and employs a dual-output architecture to balance toxicity classification with fairness considerations.

## Overview

This BiLSTM model addresses the challenge of detecting toxic comments while minimizing bias against underrepresented groups. The architecture combines collaborative filtering for user-item interactions with content-based filtering using user metadata.

### Key Features

- **Pre-trained GloVe Embeddings**: Utilizes 300-dimensional GloVe embeddings for semantic understanding
- **Dual-Output Architecture**:
  - Primary output for toxicity classification
  - Auxiliary output for subgroup identification to reduce bias
- **Advanced Pooling**: Combines Global Max Pooling and Global Average Pooling
- **Regularization**: Multiple dropout and batch normalization layers to prevent overfitting
- **Residual Connections**: Skip connections for better gradient flow

## Architecture

### Model Components

1. **Embedding Layer**
   - Initialized with pre-trained GloVe embeddings (300d)
   - Non-trainable to preserve semantic knowledge

2. **Bidirectional LSTM Layers**
   - 2 stacked BiLSTM layers with 128 units each
   - Captures both forward and backward context

3. **Pooling Layer**
   - Global Max Pooling: Captures most significant features
   - Global Average Pooling: Captures overall context
   - Concatenated for comprehensive representation

4. **Dense Layers**
   - Multiple dense layers with residual connections
   - Batch normalization and dropout for regularization

5. **Output Layers**
   - **OutputToxicity**: Binary classification (toxic/non-toxic)
   - **OutputAux**: Multi-label classification for 6 subgroups

### Model Diagram

```
Input → Embedding → SpatialDropout1D → BiLSTM(128) → BiLSTM(128) → [MaxPool, AvgPool]
                                                                           ↓
                                                                      Concatenate
                                                                           ↓
                                                    [BatchNorm → Dropout → Dense] × 3
                                                                           ↓
                                                              [OutputToxicity, OutputAux]
```

## Project Structure

```
Bi-directional-LSTM/
├── data/
│   ├── train_x.csv              # Training text data
│   ├── train_y.csv              # Training labels
│   ├── val_x.csv                # Validation text data
│   ├── val_y.csv                # Validation labels
│   ├── test_x.csv               # Test text data
│   └── glove.6B.300d.txt        # GloVe embeddings (download separately)
├── src/
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── glove_utils.py           # GloVe embedding utilities
│   ├── model.py                 # BiLSTM model architecture
│   ├── train.py                 # Training utilities
│   └── main.py                  # Main execution script
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Bi-directional-LSTM.git
cd Bi-directional-LSTM
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download GloVe embeddings**:
```bash
# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.300d.txt data/
```

## Usage

### Data Preparation

Ensure your data files are in CSV format with the following structure:

- **train_x.csv / val_x.csv / test_x.csv**: Must contain a `string` column with text data
- **train_y.csv / val_y.csv**: Must contain labels (binary or multi-label)

### Training

Run the main script to train the model:

```bash
cd src
python main.py
```

This will:
1. Load and preprocess the data
2. Create GloVe embedding matrix
3. Initialize the BiLSTM model
4. Train the model with early stopping
5. Save the best model checkpoint

### Configuration

Modify hyperparameters in `src/main.py`:

```python
# Model parameters
max_seq_length = 100      # Maximum sequence length
vocab_size = 20000        # Vocabulary size
glove_vec_dim = 300       # GloVe embedding dimension
rnn_units = 128           # LSTM units

# Training parameters
batch_size = 32           # Training batch size
epochs = 10               # Maximum epochs (early stopping applies)
learning_rate = 1e-3      # Adam optimizer learning rate
```

### Model Outputs

The model produces two outputs:

1. **Toxicity Score** (primary): Binary probability of toxic content
2. **Subgroup Scores** (auxiliary): Probabilities for 6 identity subgroups

Loss weights are configured as:
- Toxicity: 0.7
- Auxiliary: 0.3

## Training Details

- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss Function**: Binary cross-entropy for both outputs
- **Callbacks**:
  - Early stopping (patience: 3 epochs)
  - Model checkpoint (saves best model based on val_loss)
- **Regularization**:
  - Spatial Dropout (0.2)
  - Dropout layers (0.3-0.5)
  - Batch Normalization

## Results

The model achieves bias-aware toxic comment classification by:
- Using non-trainable GloVe embeddings to preserve semantic meaning
- Employing auxiliary outputs to identify and account for subgroup-related patterns
- Applying multiple regularization techniques to prevent overfitting

### Performance Considerations

- Training time depends on dataset size and GPU availability
- GPU highly recommended for faster training
- Model checkpoints are saved automatically

## Requirements

See `requirements.txt` for complete dependencies:

- tensorflow >= 2.10.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0

## GloVe Embeddings

This project uses pre-trained GloVe embeddings:

**GloVe (Global Vectors for Word Representation)**
- Dimension: 300
- Pre-trained on 6 billion tokens
- Download: [Stanford NLP](https://nlp.stanford.edu/projects/glove/)

Place `glove.6B.300d.txt` in the `data/` directory before training.

## Model Files

After training, the following files are generated:

- `model_bilstm.h5`: Best model checkpoint
- `tokenizer.pkl`: Fitted tokenizer for inference

## Citation

If you use this code, please acknowledge the following:

- GloVe embeddings: Pennington et al., 2014
- BiLSTM architecture for text classification

## License

MIT License - feel free to use this code for your projects!

## Acknowledgments

- Pre-trained GloVe embeddings by Stanford NLP
- Inspired by unbiased NLP research
- Built with TensorFlow/Keras

## Author

Ragini Chatterjee

---

**Note**: This model is designed for educational and research purposes. Ensure you have appropriate data and comply with ethical AI guidelines when deploying toxicity detection systems.
