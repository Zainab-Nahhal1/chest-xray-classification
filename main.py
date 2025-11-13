
import argparse
from source.data_loader import create_generators
from source.model import build_simple_cnn
from source.train import train
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_gen):
    y_pred = (model.predict(test_gen) > 0.5).astype('int32').flatten()
    y_true = test_gen.classes
    print('\nClassification Report:\n')
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
    cm = confusion_matrix(y_true, y_pred)
    print('\nConfusion Matrix:\n', cm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','evaluate'], default='train')
    parser.add_argument('--data_dir', default='data/chest_xray')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    train_gen, val_gen, test_gen, class_weights = create_generators(args.data_dir, batch_size=args.batch_size)
    model = build_simple_cnn()
    if args.mode == 'train':
        train(model, train_gen, val_gen, class_weights, epochs=args.epochs)
    else:
        # load best model if exists and evaluate
        try:
            model.load_weights('models/best_model.h5')
        except Exception as e:
            print('No saved model found, running evaluation with current weights. Error:', e)
        evaluate_model(model, test_gen)

if __name__ == '__main__':
    main()
