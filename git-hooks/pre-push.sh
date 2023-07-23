#!/bin/bash

set -e
poetry run build
COMMIT=$(git log --oneline | head -n 1)
cd dist/
poetry run kaggle datasets version -m "$COMMIT"
cd -