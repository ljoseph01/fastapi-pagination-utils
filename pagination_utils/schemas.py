from pydantic import BaseModel, Field
from sqlmodel import SQLModel


class PaginationQuery(BaseModel):
    """Query parameter schema for pagination queries."""

    page_index: int = Field(
        1,
        gt=0,
        description="Page index of data to retrieve. 1-indexed."
    )
    page_size: int = Field(
        10,
        gt=0,
        description="Size of pages to fetch."
    )


class PaginatedResults[Model: SQLModel](BaseModel):
    """Schema for results returned by a paginated endpoint."""

    page: int
    num_pages: int
    total_items: int
    page_size: int
    has_next: bool
    has_prev: bool
    data: list[Model]
