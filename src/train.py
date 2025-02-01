from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, train_data, train_labels, val_data, val_labels, batch_size=32, epochs=4):
    """
    Train the BiLSTM model.
    Please lower epochs if training takes forever
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('models/bilstm_best.h5', save_best_only=True, monitor='val_loss')

    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )
    return history
