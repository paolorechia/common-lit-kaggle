#!/bin/bash

# Inits Poetry
set -e
poetry run isort --check-only --profile black common_lit_kaggle
poetry run black --check common_lit_kaggle
poetry run pylint common_lit_kaggle
poetry run mypy common_lit_kaggle