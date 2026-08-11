from pathlib import Path
import shutil
import sys
import traceback
from threading import Lock

from fastapi import FastAPI, Response

sys.path.insert(0, "/opt/algorithm/src")
from process import main as run_process  # noqa: E402


app = FastAPI()
_lock = Lock()


@app.get("/health")
def health():
    model_dir = Path("/opt/ml/model")
    required = [
        model_dir / "checkpoint_best.pth",
        model_dir / "dataset.json",
        model_dir / "nnUNetResEncUNetLPlans.json",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return Response(
            content="Missing model files: " + ", ".join(missing),
            status_code=503,
            media_type="text/plain",
        )

    return Response(content="OK", status_code=200, media_type="text/plain")


@app.post("/invoke")
def invoke():
    with _lock:
        try:
            # Grand Challenge calls /invoke after mounting /input and /output.
            # Clear only our temporary workspace, never /input or /output.
            work_dir = Path("/tmp/selma3d_nnunet_work")
            if work_dir.exists():
                shutil.rmtree(work_dir)

            run_process()

            return Response(
                content="Prediction completed",
                status_code=201,
                media_type="text/plain",
            )
        except Exception:
            traceback.print_exc()
            return Response(
                content=traceback.format_exc(),
                status_code=500,
                media_type="text/plain",
            )
