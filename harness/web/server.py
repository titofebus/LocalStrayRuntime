"""FastAPI web server for visual side-by-side evaluation comparison."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from harness.core.recorder import ResultsRecorder
from harness.core.suites import SuiteLoader

app = FastAPI(title="Local vs Frontier Model Arena")

WEB_DIR = Path(__file__).resolve().parent

@app.get("/api/challenges")
def api_challenges():
    return [c.model_dump() for c in SuiteLoader.load_all_challenges()]

@app.get("/api/results")
def api_results():
    return [r.model_dump() for r in ResultsRecorder.load_all_records()]

@app.get("/api/leaderboard")
def api_leaderboard():
    return ResultsRecorder.compute_leaderboard()

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = WEB_DIR / "index.html"
    return FileResponse(index_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
