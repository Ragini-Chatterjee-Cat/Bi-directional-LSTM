from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, train_padded, train_labels, val_padded, val_labels, batch_size=32, epochs=10):
    """
    Compiles and trains the BiLSTM model.
    """
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('model_bilstm.h5', save_best_only=True, monitor='val_loss', mode='min')

    history = model.fit(
        train_padded, train_labels,
        validation_data=(val_padded, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint]
    )

    return history
