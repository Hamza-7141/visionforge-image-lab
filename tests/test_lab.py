"""Offline regressions using synthetic colored images, never personal files."""
import io
import os
from pathlib import Path
import sys
import time
import unittest
import uuid
import zipfile
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app import app, SESSIONS, session, expire_sessions, TTL
from utils import load_dataset, preprocess_image_bytes
from trainers import train_all_models, export_models_to_onnx
from inference import predict_all_models

def picture(color):
    output = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(output, format='PNG')
    return output.getvalue()

class LabTests(unittest.TestCase):
    def setUp(self):
        SESSIONS.clear()
        self.client = TestClient(app)
        self.headers = {'X-Session-Id': str(uuid.uuid4())}

    def test_health_and_ui(self):
        self.assertEqual(self.client.get('/health').json(), {'status':'ok'})
        self.assertIn('VisionForge', self.client.get('/').text)
        self.assertEqual(self.client.get('/').headers['Cache-Control'], 'no-store')

    def test_required_session_and_bad_image(self):
        self.assertEqual(self.client.get('/dataset').status_code, 400)
        response=self.client.post('/upload', headers=self.headers, data={'class_name':'red'}, files={'file':('bad.png',b'bad','image/png')})
        self.assertEqual(response.status_code, 400)

    def test_class_path_and_upload_limit(self):
        response=self.client.post('/upload', headers=self.headers, data={'class_name':'../outside'}, files={'file':('x.png',picture('red'),'image/png')})
        self.assertEqual(response.status_code, 400)
        with self.assertRaises(ValueError): preprocess_image_bytes(b'x'*(3*1024*1024+1))

    def test_isolation_reset_and_expiry(self):
        response=self.client.post('/upload', headers=self.headers, data={'class_name':'red'}, files={'file':('x.png',picture('red'),'image/png')})
        self.assertEqual(response.status_code, 200)
        other={'X-Session-Id':str(uuid.uuid4())}
        self.assertEqual(self.client.get('/dataset',headers=other).json(), {})
        self.assertEqual(self.client.get('/dataset',headers=self.headers).json(), {'red':1})
        self.assertEqual(self.client.delete('/session',headers=self.headers).status_code,200)
        self.assertEqual(self.client.get('/dataset',headers=self.headers).json(),{})
        item=session(self.headers['X-Session-Id']); item['touched']=time.monotonic()-TTL-1
        expire_sessions(); self.assertNotIn(self.headers['X-Session-Id'],SESSIONS)

    def test_invalid_training_and_prediction(self):
        self.assertEqual(self.client.post('/train',headers=self.headers).status_code,400)
        self.assertEqual(self.client.post('/predict/upload',headers=self.headers,files={'file':('x.png',picture('red'),'image/png')}).status_code,400)

    def test_split_and_three_real_models_and_export(self):
        data={name:[preprocess_image_bytes(picture(color)) for _ in range(4)] for name,color in [('red','red'),('blue','blue')]}
        x,v,y,vy,labels=load_dataset(data)
        self.assertEqual(set(y),set(vy)); self.assertEqual(len(x)+len(v),8)
        models,labels,results=train_all_models(data,{},epochs=1)
        self.assertEqual(set(models),{'logreg','rf','cnn'})
        predictions=predict_all_models(x[0],models,labels)
        self.assertEqual(set(predictions),set(models))
        for result in predictions.values(): self.assertAlmostEqual(sum(result['probs']),1,places=5)
        for result in results.values(): self.assertTrue(0<=result['accuracy']<=1)
        with zipfile.ZipFile(io.BytesIO(export_models_to_onnx(models,labels))) as archive:
            self.assertIn('logreg.onnx',archive.namelist()); self.assertIn('rf.onnx',archive.namelist())

if __name__ == '__main__': unittest.main()
