import pathlib
from dataclasses import dataclass

from common_lit_kaggle.framework.schema import Schema


@dataclass
class TableReference:
    name: str
    path: pathlib.Path
    schema: type[Schema]
    format: str
