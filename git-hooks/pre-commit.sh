#!/bin/bash

# Inits Poetry
set -e
poetry run isort --check-only --profile black src
poetry run black --check src
poetry run pylint src
poetry run mypy src