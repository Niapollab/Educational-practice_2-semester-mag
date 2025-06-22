#!/usr/bin/env python3
from datetime import timedelta
from typing import Any
from flask import Flask, session, redirect, request, redirect
from uuid import uuid4
import os
import requests


model_endpoint = os.environ.get("MODEL_ENDPOINT")
if not model_endpoint:
    raise ValueError("MODEL_ENDPOINT environment variable must be set.")


app = Flask(__name__, static_url_path="")
app.secret_key = os.urandom(64).hex()
app.permanent_session_lifetime = timedelta(hours=6)


@app.route("/", methods=["GET"])
def index() -> Any:
    if "uid" not in session:
        session["uid"] = str(uuid4())

    return app.send_static_file("index.html")


@app.route("/", methods=["POST"])
def make_turn() -> Any:
    if "uid" not in session:
        return redirect("/")

    return requests.post(
        model_endpoint, json=request.json, headers={"UID": session["uid"]}
    ).json()


def main() -> None:
    app.run(host="0.0.0.0", port=80)


if __name__ == "__main__":
    main()
