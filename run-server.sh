#!/bin/sh

export FLASK_APP=server.py

cd dist-model-server &&
python -m flask run