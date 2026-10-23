"""VisionForge: session-isolated, memory-only image classification lab."""
import copy
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from utils import MAX_BYTES, preprocess_image_bytes, load_dataset
from trainers import train_all_models, export_models_to_onnx
from inference import predict_all_models

SESSIONS = {}
LOCK = threading.RLock()
TRAINING_LOCK = threading.Lock()
TTL = 20 * 60

def expire_sessions():
    now = time.monotonic()
    for key in list(SESSIONS):
        item = SESSIONS[key]
        if now-item['touched'] > TTL and item['state']['status'] not in ('queued', 'training'):
            del SESSIONS[key]

@asynccontextmanager
async def lifespan(app):
    stop = threading.Event()
    def janitor():
        while not stop.wait(30):
            with LOCK:
                expire_sessions()
    worker = threading.Thread(target=janitor, daemon=True)
    worker.start()
    yield
    stop.set()
    worker.join(timeout=1)
    with LOCK:
        SESSIONS.clear()

app = FastAPI(title='VisionForge — Image Learning Lab', lifespan=lifespan)

class UploadTooLarge(Exception):
    pass

class UploadLimit:
    """Bound multipart bodies below the parser's disk-spooling threshold."""
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        total = 0
        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > MAX_BYTES + 16384:
                raise UploadTooLarge()
            return message
        try:
            await self.app(scope, bounded_receive, send)
        except UploadTooLarge:
            await JSONResponse({'detail': 'Upload exceeds the 512 KB image limit.'}, status_code=413)(scope, receive, send)

app.add_middleware(UploadLimit)

def session(session_id):
    if not session_id or not re.fullmatch(r'[a-f0-9-]{36}', session_id):
        raise HTTPException(400, 'A valid temporary session is required. Refresh the page.')
    with LOCK:
        now = time.monotonic()
        expire_sessions()
        if session_id not in SESSIONS:
            if len(SESSIONS) >= 12:
                raise HTTPException(503, 'The demo is busy. Try again later.')
            SESSIONS[session_id] = {'data': {}, 'models': {}, 'labels': {}, 'state': {'status': 'idle'}, 'touched': now}
        SESSIONS[session_id]['touched'] = now
        return SESSIONS[session_id]

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/dataset')
def dataset(x_session_id: str = Header(default='')):
    with LOCK:
        item = session(x_session_id)
        return {name: len(images) for name, images in item['data'].items()}

@app.post('/upload')
async def upload(class_name: str = Form(...), file: UploadFile = File(...), x_session_id: str = Header(default='')):
    name = class_name.strip()
    if not re.fullmatch(r'[\w -]{1,40}', name):
        raise HTTPException(400, 'Use 1–40 letters, numbers, spaces, underscores or hyphens for a class.')
    raw = await file.read(MAX_BYTES+1)
    try:
        image = preprocess_image_bytes(raw)
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    with LOCK:
        item = session(x_session_id)
        if item['state']['status'] in ('queued', 'training'):
            raise HTTPException(409, 'Wait for training before changing the dataset.')
        data = item['data']
        if (name not in data and len(data) >= 6) or sum(map(len, data.values())) >= 120:
            raise HTTPException(400, 'Demo limit: six classes and 120 images per session.')
        data.setdefault(name, []).append(image)
        item['models'] = {}
        item['state'] = {'status': 'idle'}
        return {'message': 'Image added in memory', 'classes': {k: len(v) for k, v in data.items()}}

def training_job(item):
    try:
        models, labels, results = train_all_models(item['data'], item['state'], state_lock=LOCK)
        with LOCK:
            item.update(models=models, labels=labels)
            item['state'].update(status='finished', results=results)
    except Exception:
        item['state'].update(status='error', error='Training failed. Check that your dataset contains varied, valid images and try a smaller dataset.')
    finally:
        TRAINING_LOCK.release()

@app.post('/train')
def train(tasks: BackgroundTasks, x_session_id: str = Header(default='')):
    with LOCK:
        item = session(x_session_id)
        if item['state']['status'] in ('queued', 'training'):
            raise HTTPException(409, 'Training is already running.')
        try:
            load_dataset(item['data'])
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        if not TRAINING_LOCK.acquire(blocking=False):
            raise HTTPException(503, 'Another training job is running. Try again shortly.')
        item['models'] = {}
        item['state'] = {'status': 'queued'}
        tasks.add_task(training_job, item)
        return {'message': 'Training started'}

@app.get('/training/status')
def status(x_session_id: str = Header(default='')):
    with LOCK:
        return copy.deepcopy(session(x_session_id)['state'])

@app.post('/predict/upload')
async def predict(file: UploadFile = File(...), x_session_id: str = Header(default='')):
    item = session(x_session_id)
    with LOCK:
        models, labels = item['models'], item['labels']
    if not models:
        raise HTTPException(400, 'Train your dataset before making predictions.')
    try:
        image = preprocess_image_bytes(await file.read(MAX_BYTES+1))
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    return predict_all_models(image, models, labels)

@app.post('/export/onnx')
def export(x_session_id: str = Header(default='')):
    item = session(x_session_id)
    if not item['models']:
        raise HTTPException(400, 'Train first, then export.')
    return Response(export_models_to_onnx(item['models'], item['labels']), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename=visionforge-onnx.zip'})

@app.delete('/session')
def reset(x_session_id: str = Header(default='')):
    with LOCK:
        item = session(x_session_id)
        if item['state']['status'] in ('queued', 'training'):
            raise HTTPException(409, 'Wait for training to finish before resetting.')
        del SESSIONS[x_session_id]
    return {'message': 'Temporary dataset and models removed.'}

@app.middleware('http')
async def privacy_headers(request, call_next):
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Permissions-Policy'] = 'camera=(self), microphone=()'
    return response

app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[1]/'frontend', html=True), name='frontend')
