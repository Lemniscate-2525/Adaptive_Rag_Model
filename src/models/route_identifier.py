"""Validated route choices produced by the query classifier."""

from typing import Literal
from pydantic import BaseModel


class RouteIdentifier(BaseModel):
    """The classifier must choose one supported path."""

    route: Literal["index", "general", "search"]
