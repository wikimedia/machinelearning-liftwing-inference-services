#!/bin/bash

set -ex

# Run the server
exec /usr/bin/python3 -m uvicorn tts_generator.service:app --host 0.0.0.0 --port 8080 --workers 1
