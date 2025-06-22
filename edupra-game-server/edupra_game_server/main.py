#!/usr/bin/env python3
from datetime import timedelta
from typing import Any, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer
from uuid import uuid4
import os
import requests
import uvicorn


model_endpoint = os.environ.get("MODEL_ENDPOINT")
if not model_endpoint:
    raise ValueError("MODEL_ENDPOINT environment variable must be set.")


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
secret_key = os.urandom(64).hex()
serializer = URLSafeSerializer(secret_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index(request: Request) -> Any:
    session_data = request.cookies.get("session")

    if session_data:
        try:
            session = serializer.loads(session_data)
        except Exception:
            session = {}
    else:
        session = {}

    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    response = FileResponse(static_file)

    if "uid" not in session:
        session["uid"] = str(uuid4())
        response.set_cookie(
            key="session",
            value=serializer.dumps(session),
            max_age=6 * 60 * 60,
            httponly=True,
            secure=False,
        )

    return response


@app.post("/")
async def make_turn(request: Request, request_data: Dict[str, Any]) -> Any:
    session_data = request.cookies.get("session")

    if not session_data:
        raise HTTPException(status_code=401, detail="Session not found")

    try:
        session = serializer.loads(session_data)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

    if "uid" not in session:
        raise HTTPException(status_code=401, detail="UID not found in session")

    response = requests.post(
        model_endpoint, json=request_data, headers={"UID": session["uid"]}
    )

    return response.json()


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=80)


if __name__ == "__main__":
    main()
