#!/bin/bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export ES_PATH_CONF=/opt/homebrew/etc/elasticsearch
/opt/homebrew/bin/elasticsearch "$@"
