#!/bin/bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
nohup /opt/homebrew/bin/elasticsearch -Epath.data=./tmp/elastic/data -Epath.logs=./tmp/elastic/logs -Expack.ml.enabled=false > ./tmp/elastic/logs/stdout.log 2>&1 &
echo "Elasticsearch started in the background. Check logs at ./tmp/elastic/logs/"
