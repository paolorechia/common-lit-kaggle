#!/bin/bash

# Inits Poetry
poetry shell

isort --profile black src
black --check src
pylint src
mypy src