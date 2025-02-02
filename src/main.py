from data_utils import load_data, preprocess_labels, tokenize_and_pad_sequences
from glove_utils import create_glove_embedding_matrix
from model import getModelBiLSTM
from train import train_model
import os

def main():
    # Constants
    max_seq_length = 100
    vocab_size = 20000
    glove_vec_dim = 300
    glove_file = '../data/glove.6B.300d.txt'
    batch_size = 32
    epochs = 10
    model_save_path = '../results/model_bilstm.h5'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_x, train_y, val_x, val_y, test_x = load_data('../data')  # Adjust path to match your data folder
    train_labels, val_labels = preprocess_labels(train_y, val_y)

    # Tokenize and pad sequences
    print("Tokenizing and padding sequences...")
    train_padded, val_padded, test_padded, tokenizer = tokenize_and_pad_sequences(
        train_x, val_x, test_x, max_seq_length, vocab_size
    )

    # Create GloVe embedding matrix
    print("Creating GloVe embedding matrix...")
    embedding_matrix = create_glove_embedding_matrix(tokenizer, glove_file, vocab_size, glove_vec_dim)

    # Initialize the model
    print("Initializing BiLSTM model...")
    model = getModelBiLSTM(
        seqLen=max_seq_length,
        vocabSize=vocab_size,
        gloveVecDim=glove_vec_dim,
        weightMatrix=embedding_matrix,
        isTrainable=False,
        rnnUnits=128
    )

    # Train the model
    print("Training the model...")
    history = train_model(
        model=model,
        train_padded=train_padded,
        train_labels=train_labels,
        val_padded=val_padded,
        val_labels=val_labels,
        batch_size=batch_size,
        epochs=epochs,
        model_save_path=model_save_path
    )

    # Save tokenizer for future use
    tokenizer_path = '../results/tokenizer.pkl'
    print(f"Saving tokenizer to {tokenizer_path}...")
    with open(tokenizer_path, 'wb') as f:
        import pickle
        pickle.dump(tokenizer, f)

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
