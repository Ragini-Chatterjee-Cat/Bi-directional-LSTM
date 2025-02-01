import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os



def load_data():
    """
    Load training, validation, and test datasets from CSV files located in ../data.
    """
    # Relative paths to the data directory
    train_x_path = "../data/train_x.csv"
    train_y_path = "../data/train_y.csv"
    val_x_path = "../data/val_x.csv"
    val_y_path = "../data/val_y.csv"

    # Load datasets
    train_x = pd.read_csv(train_x_path)['string'].values
    train_y = pd.read_csv(train_y_path)['y'].values
    val_x = pd.read_csv(val_x_path)['string'].values
    val_y = pd.read_csv(val_y_path)['y'].values



def preprocess_texts(texts, tokenizer, max_seq_len=100):
    """
    Tokenize and pad sequences.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_seq_len, padding='post', truncating='post')
    return padded

def create_tokenizer(texts, vocab_size=20000):
    """
    Create and fit a tokenizer on the given texts.
    """
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    return tokenizer
