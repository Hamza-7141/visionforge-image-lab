"""Three real classifiers, held in memory for one visitor's session."""
import io
import json
import zipfile
from contextlib import nullcontext
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import load_dataset

def train_all_models(dataset, state, epochs=4, state_lock=None):
    def publish(**changes):
        with state_lock if state_lock is not None else nullcontext():
            state.update(changes)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    x, v, y, vy, labels = load_dataset(dataset)
    publish(status='training', device={'gpu_available': False, 'num_gpus': 0, 'gpus': []})
    models, results = {}, {}
    flat_x, flat_v = x.reshape(len(x), -1), v.reshape(len(v), -1)
    for key, classifier, name in [
        ('logreg', LogisticRegression(max_iter=300), 'Logistic Regression'),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1), 'Random Forest')]:
        publish(stage=name)
        classifier.fit(flat_x, y)
        prediction = classifier.predict(flat_v)
        models[key] = classifier
        results[key] = {'name': name, 'accuracy': float(accuracy_score(vy, prediction)),
                        'confusion_matrix': confusion_matrix(vy, prediction, labels=list(labels)).tolist()}
    publish(stage='CNN')
    cnn = tf.keras.Sequential([
        tf.keras.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(labels), activation='softmax')])
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    # Explicit batches avoid background tf.data threads on small cloud CPUs.
    for epoch in range(epochs):
        cnn.reset_metrics()
        for start in range(0, len(x), 16):
            cnn.train_on_batch(x[start:start+16], y[start:start+16])
        train_prob = cnn(x, training=False).numpy()
        val_prob = cnn(v, training=False).numpy()
        history['accuracy'].append(float(accuracy_score(y, train_prob.argmax(axis=1))))
        history['val_accuracy'].append(float(accuracy_score(vy, val_prob.argmax(axis=1))))
        history['loss'].append(float(-np.log(np.clip(train_prob[np.arange(len(y)), y], 1e-7, 1)).mean()))
        history['val_loss'].append(float(-np.log(np.clip(val_prob[np.arange(len(vy)), vy], 1e-7, 1)).mean()))
        publish(cnn={'current_epoch': epoch+1, 'total_epochs': epochs,
                        **{key: values[-1] for key, values in history.items()}})
    models['cnn'] = cnn
    pred = val_prob.argmax(axis=1)
    results['cnn'] = {'name': 'CNN', 'accuracy': float(accuracy_score(vy, pred)),
                      'confusion_matrix': confusion_matrix(vy, pred, labels=list(labels)).tolist(), 'history': history}
    for result in results.values():
        result.update(training_images=len(x), validation_images=len(v))
    return models, labels, results

def export_models_to_onnx(models, labels):
    """Download classical ONNX models in a ZIP, without saving model files."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for key in ('logreg', 'rf'):
            model = convert_sklearn(models[key], initial_types=[('input', FloatTensorType([None, 12288]))])
            archive.writestr(key+'.onnx', model.SerializeToString())
        archive.writestr('labels.json', json.dumps(labels))
        archive.writestr('README.txt', 'Classical models only. Input: RGB 64x64, normalized to [0,1], flattened float32. CNN export is not supported in this release.')
    return buffer.getvalue()
