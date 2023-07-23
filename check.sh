#!/bin/bash
poetry run isort --profile black common_lit_kaggle
poetry run black common_lit_kaggle
poetry run pylint common_lit_kaggle
poetry run mypy common_lit_kaggle