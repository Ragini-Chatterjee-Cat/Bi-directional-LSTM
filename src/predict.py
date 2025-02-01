import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

def make_predictions(model, test_texts, tokenizer, max_seq_len=100):
    """
    Make predictions on test data.
    """
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_padded = pad_sequences(test_sequences, maxlen=max_seq_len, padding='post')
    predictions = model.predict(test_padded)
    return predictions

def save_predictions(predictions, test_ids, output_file="submission.csv"):
    """
    Save predictions to a CSV file for Kaggle submission.
    """
    submission = pd.DataFrame({'ID': test_ids, 'Prediction': predictions.flatten()})
    submission.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
