
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json

def train(model, train_gen, val_gen, class_weights, epochs=10, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    ckpt = ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), save_best_only=True, monitor='val_loss')
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, class_weight=class_weights, callbacks=[ckpt, es])
    # save history
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)
    return history
