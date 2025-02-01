import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate,
    Dense, Dropout, BatchNormalization, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend

def getModelBiLSTM(seq_len, vocab_size, glove_vec_dim, weight_matrix, rnn_units=128):
    """
    Define the BiLSTM model architecture.
    """
    backend.clear_session()
    input_layer = Input(shape=(seq_len,))
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=glove_vec_dim,
        weights=[weight_matrix],
        trainable=False
    )(input_layer)

    x = Bidirectional(LSTM(rnn_units, return_sequences=True))(embedding_layer)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    combined = Concatenate()([max_pool, avg_pool])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
