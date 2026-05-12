from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Point(BaseModel):
    x: int
    y: int


class WallModel(BaseModel):
    id: str
    start: Point
    end: Point


class OpeningModel(BaseModel):
    id: str
    wallId: str
    offset: float
    width: float
    type: str


class ProcessFloorplanResponse(BaseModel):
    walls: list[WallModel]
    openings: list[OpeningModel]

    model_config = ConfigDict(extra="forbid")
