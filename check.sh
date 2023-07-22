#!/bin/bash
poetry run isort --profile black src
poetry run black src
poetry run pylint src
poetry run mypy src