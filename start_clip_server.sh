#!/bin/bash

export JINA_LOG_LEVEL=DEBUG && source .venv/bin/activate && python -m clip_server search_flow.yaml