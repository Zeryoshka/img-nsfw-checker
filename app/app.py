from typing import Union
from typing import Optional
from flask import Flask
from flask import request
from fastapi import FastAPI, UploadFile, File
from app.flow import build_flow, NSFWTask
import logging
import asyncio

ALLOWED_IMG_CONTENT_TYPES = ['image/jpeg']

loop = asyncio.get_event_loop()
nsfw_flow = build_flow()
nsfw_flow.start()
app = Flask(__name__)
log = logging.getLogger()


@app.get("/")
def read_root():
    from app import __version__
    return {"version": __version__}


@app.post("/check-nsfw")
def read_item(file: UploadFile = File(...)):
    request.content_type
    file = request.files['img_to_check']
    if file.content_type not in ALLOWED_IMG_CONTENT_TYPES:
        log.warning('gotten file type %s is not allowed', file.content_type)
        return {}

    content = file.read()

    task = NSFWTask(content)
    loop.run_until_complete(
        nsfw_flow.process(task),
    )
    return task.img_class

