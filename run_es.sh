#!/bin/bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
/opt/homebrew/bin/elasticsearch -Epath.data=./tmp/elastic/data -Epath.logs=./tmp/elastic/logs -Expack.ml.enabled=false
