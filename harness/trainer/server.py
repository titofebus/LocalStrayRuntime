"""FastAPI server for DFlash Training & Progress Utility Web App."""
import asyncio
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from harness.trainer.orchestrator import orchestrator

app = FastAPI(title="DFlash Training & Progress Utility")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static directory for CSS, JS, and Assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

active_websockets: list[WebSocket] = []


def require_experimental_training() -> None:
    if os.environ.get("QWEN_PRIME_ENABLE_EXPERIMENTAL_TRAINING") != "1":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "The synthetic training prototype is disabled. Set "
                "QWEN_PRIME_ENABLE_EXPERIMENTAL_TRAINING=1 only for local "
                "development experiments."
            ),
        )

def ws_broadcaster(data: dict):
    """Broadcast state to all connected websocket clients."""
    msg = json.dumps(data)
    for ws in list(active_websockets):
        try:
            asyncio.create_task(ws.send_text(msg))
        except Exception:
            if ws in active_websockets:
                active_websockets.remove(ws)

orchestrator.add_listener(ws_broadcaster)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DFlash Utility frontend building...</h1>"

@app.get("/api/state")
async def get_state():
    return orchestrator.state.model_dump()

@app.post("/api/start")
async def post_start():
    require_experimental_training()
    orchestrator.start_training()
    return {"status": "started"}

@app.post("/api/pause")
async def post_pause():
    require_experimental_training()
    orchestrator.pause_training()
    return {"status": "paused"}

@app.post("/api/reset")
async def post_reset():
    require_experimental_training()
    orchestrator.reset_training()
    return {"status": "reset"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if os.environ.get("QWEN_PRIME_ENABLE_EXPERIMENTAL_TRAINING") != "1":
        await websocket.close(code=1008, reason="Experimental training is disabled")
        return
    await websocket.accept()
    active_websockets.append(websocket)
    # Send initial state
    await websocket.send_text(json.dumps(orchestrator.state.model_dump()))
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data).get("action")
            if cmd == "start":
                orchestrator.start_training()
            elif cmd == "pause":
                orchestrator.pause_training()
            elif cmd == "reset":
                orchestrator.reset_training()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("harness.trainer.server:app", host="127.0.0.1", port=8766, reload=False)
