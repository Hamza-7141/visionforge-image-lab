"""Predict using this session's models, avoiding stale disk caches."""
import numpy as np

def predict_all_models(image_array, models, labels):
    batch = image_array[np.newaxis, ...]
    results = {}
    for key, model in models.items():
        if key == 'cnn':
            probabilities = model(batch, training=False).numpy()[0]
            index = int(probabilities.argmax())
        else:
            probabilities = model.predict_proba(batch.reshape(1, -1))[0]
            index = int(model.classes_[probabilities.argmax()])
        results[key] = {'class_index': index, 'class_name': labels[index], 'probs': probabilities.tolist()}
    return results
