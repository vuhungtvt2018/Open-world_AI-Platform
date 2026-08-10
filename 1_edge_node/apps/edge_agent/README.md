# Refactored Visual Inspection Backend

The original backend is split by responsibility while preserving the existing processing logic.

## File structure

- `web_backend.py`: creates the FastAPI application, mounts static directories, and includes routers
- `app_config.py`: resolves the project root and initializes shared configuration and directories
- `lifecycle.py`: initializes the database, seeds model records, starts cleanup, and registers the EventBus callback
- `inference.py`: contains `WebInference` and the shared inference engine
- `routers/inspection.py`: upload, camera stream, inspection, history, and inspection image APIs
- `routers/monitoring.py`: dataset, model registry, system health, and analytics APIs
- `routers/synchronization.py`: cloud-edge synchronization APIs
- `routers/reports.py`: report lookup and download APIs

## Placement

Place the `refactored_web_backend` directory inside `apps/edge-agent`, next to the original `web_backend.py`.

## Run

Run directly from `apps/edge-agent`:

```bash
python refactored_web_backend/web_backend.py
```

Alternatively, run with Uvicorn from `apps/edge-agent`:

```bash
uvicorn refactored_web_backend.web_backend:app --host 0.0.0.0 --port 8000
```
