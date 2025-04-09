#!/bin/bash
cd "$(dirname "$0")"
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export ES_PATH_CONF=/opt/homebrew/etc/elasticsearch
export ES_HOME=/opt/homebrew/Cellar/elasticsearch-full/7.17.4/libexec
python scripts/es_health_check.py
