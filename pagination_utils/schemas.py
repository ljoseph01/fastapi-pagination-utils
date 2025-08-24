from pydantic import BaseModel
from sqlmodel import SQLModel


class PaginatedResults[Model: SQLModel](BaseModel):
    page: int
    num_pages: int
    total_items: int
    page_size: int
    has_next: bool
    has_prev: bool
    data: list[Model]
