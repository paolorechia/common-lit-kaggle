#!/bin/bash

# Inits Poetry
poetry run isort --profile black src
poetry run black --check src
poetry run pylint src
poetry run mypy src