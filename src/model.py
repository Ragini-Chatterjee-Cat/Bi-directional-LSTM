import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, SpatialDropout1D, Bidirectional, LSTM,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate,
    BatchNormalization, Dropout, Dense, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend


def getModelBiLSTM(seqLen, vocabSize, gloveVecDim, weightMatrix, isTrainable=False, rnnUnits=128):
    """
    Defines the BiLSTM model architecture for toxic comment classification.

    Parameters:
    ----------
    seqLen : int
        Maximum sequence length for the input text.
    vocabSize : int
        Vocabulary size for the tokenizer.
    gloveVecDim : int
        Dimension of the GloVe embeddings.
    weightMatrix : numpy.ndarray
        Pre-trained embedding matrix initialized with GloVe embeddings.
    isTrainable : bool
        Whether to fine-tune the embedding layer during training.
    rnnUnits : int
        Number of units in the Bidirectional LSTM layers.

    Returns:
    -------
    model : tensorflow.keras.Model
        Compiled Keras model.
    """
    # Clear any previous models in the backend
    backend.clear_session()

    # Input layer
    inputLayerText = Input(shape=(seqLen,), name='InputLayerText')

    # Embedding layer
    embeddingLayerText = Embedding(
        input_dim=vocabSize,
        output_dim=gloveVecDim,
        weights=[weightMatrix],
        trainable=isTrainable,
        name='EmbeddingLayerText'
    )(inputLayerText)

    # Spatial Dropout layer
    spatialDropout1DLayer = SpatialDropout1D(rate=0.2, name='SpatialDropout1D')(embeddingLayerText)

    # Bidirectional LSTM layers
    biLSTMLayer1 = Bidirectional(
        LSTM(rnnUnits, return_sequences=True, name='BiLSTM1')
    )(spatialDropout1DLayer)
    biLSTMLayer2 = Bidirectional(
        LSTM(rnnUnits, return_sequences=True, name='BiLSTM2')
    )(biLSTMLayer1)

    # Global Max and Average Pooling layers
    globalMaxPooling1D = GlobalMaxPooling1D(name='GlobalMaxPooling1D')(biLSTMLayer2)
    globalAvgPooling1D = GlobalAveragePooling1D(name='GlobalAvgPooling1D')(biLSTMLayer2)

    # Concatenate pooling outputs
    concateGMaxGAvg = Concatenate(axis=1, name='ConcatenateLayerGMaxGAvg')([globalMaxPooling1D, globalAvgPooling1D])

    # Batch Normalization and Dropout layers
    batchNormLayer1 = BatchNormalization(name='BatchNormalizationLayer1')(concateGMaxGAvg)
    dropoutLayer1 = Dropout(rate=0.5, name='DropoutLayer1')(batchNormLayer1)

    # Dense and Add layers
    denseLayer1 = Dense(rnnUnits * 4, activation='relu', name='DenseLayer1')(dropoutLayer1)
    addLayer1 = Add(name='AddLayer1')([denseLayer1, concateGMaxGAvg])

    # More Batch Normalization, Dropout, Dense, and Add layers
    batchNormLayer2 = BatchNormalization(name='BatchNormalizationLayer2')(addLayer1)
    dropoutLayer2 = Dropout(rate=0.4, name='DropoutLayer2')(batchNormLayer2)
    denseLayer2 = Dense(rnnUnits * 4, activation='relu', name='DenseLayer2')(dropoutLayer2)
    addLayer2 = Add(name='AddLayer2')([denseLayer2, addLayer1])

    # Final Batch Normalization and Dropout layers
    batchNormLayer3 = BatchNormalization(name='BatchNormalizationLayer3')(addLayer2)
    dropoutLayer3 = Dropout(rate=0.3, name='DropoutLayer3')(batchNormLayer3)

    # Output layers
    outputToxicity = Dense(1, activation='sigmoid', name='OutputToxicity')(dropoutLayer3)
    outputAux = Dense(6, activation='sigmoid', name='OutputAux')(dropoutLayer3)

    # Define the model
    model = Model(inputs=inputLayerText, outputs=[outputToxicity, outputAux])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={'OutputToxicity': 'binary_crossentropy', 'OutputAux': 'binary_crossentropy'},
        loss_weights={'OutputToxicity': 0.7, 'OutputAux': 0.3},
        metrics=['accuracy']
    )

    return model
