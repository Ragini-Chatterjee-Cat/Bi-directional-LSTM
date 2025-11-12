"""
Data loading and preprocessing utilities for BiLSTM toxic comment classification.
"""
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(data_dir='../data'):
    """
    Loads train, validation, and test datasets from CSV files.

    Args:
        data_dir (str): Directory containing the data files

    Returns:
        tuple: (train_x, train_y, val_x, val_y, test_x)
    """
    train_x = pd.read_csv(f'{data_dir}/train_x.csv')
    train_y = pd.read_csv(f'{data_dir}/train_y.csv')
    val_x = pd.read_csv(f'{data_dir}/val_x.csv')
    val_y = pd.read_csv(f'{data_dir}/val_y.csv')
    test_x = pd.read_csv(f'{data_dir}/test_x.csv')

    return (
        train_x['string'].fillna("").tolist(),
        train_y,
        val_x['string'].fillna("").tolist(),
        val_y,
        test_x['string'].fillna("").tolist()
    )


def preprocess_labels(train_y, val_y):
    """
    Preprocesses labels for training.

    Args:
        train_y: Training labels DataFrame
        val_y: Validation labels DataFrame

    Returns:
        tuple: (train_labels, val_labels) as numpy arrays
    """
    train_labels = train_y.fillna(0).astype(int).values
    val_labels = val_y.fillna(0).astype(int).values
    return train_labels, val_labels


def load_and_preprocess_data():
    """
    Loads and preprocesses the dataset for training.
    """
    # Load data
    train_x = pd.read_csv('../data/train_x.csv')
    train_y = pd.read_csv('../data/train_y.csv')
    val_x = pd.read_csv('../data/val_x.csv')
    val_y = pd.read_csv('../data/val_y.csv')
    test_x = pd.read_csv('../data/test_x.csv')

    # Ensure required columns exist
    for df, name in [(train_x, "train_x"), (val_x, "val_x"), (test_x, "test_x")]:
        if 'string' not in df.columns:
            raise ValueError(f"Column 'string' is missing from {name}")

    if 'y' not in train_y.columns or 'y' not in val_y.columns:
        raise ValueError("Column 'y' is missing from the labels file (train_y/val_y).")

    # Preprocess text data
    train_data = train_x['string'].fillna("").tolist()
    val_data = val_x['string'].fillna("").tolist()
    test_data = test_x['string'].fillna("").tolist()

    # Preprocess labels
    train_labels = train_y.fillna(0).astype(int).values
    val_labels = val_y.fillna(0).astype(int).values

    return train_data, train_labels, val_data, val_labels, test_data

def tokenize_and_pad_sequences(train_data, val_data, test_data, max_seq_length=100, vocab_size=20000):
    """
    Tokenizes and pads sequences for the model.
    """
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_data)

    train_sequences = tokenizer.texts_to_sequences(train_data)
    val_sequences = tokenizer.texts_to_sequences(val_data)
    test_sequences = tokenizer.texts_to_sequences(test_data)

    train_padded = pad_sequences(train_sequences, maxlen=max_seq_length, padding='post', truncating='post')
    val_padded = pad_sequences(val_sequences, maxlen=max_seq_length, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_seq_length, padding='post', truncating='post')

    return train_padded, val_padded, test_padded, tokenizer
