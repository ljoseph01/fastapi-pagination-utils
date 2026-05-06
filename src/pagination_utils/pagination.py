import logging
import traceback
from collections.abc import Callable
from typing import Annotated, NamedTuple, Any

from fastapi import HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import Row, Select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import func, select
from sqlmodel.sql.expression import SelectOfScalar

from .schemas import PaginatedResults

logger = logging.getLogger(__name__)


class PaginationDetails(NamedTuple):
    """Details to be returned alongside paginated data."""
    num_pages: int
    has_next: bool
    has_prev: bool
    offset: int


PageIndexQuery = Annotated[int, Query(
    description="Page index of data to retrieve. 1-indexed."
)]
PageSizeQuery = Annotated[int, Query(
    description="Size of pages to fetch."
)]


def get_pagination_details(
    total_items: int,
    page: int,
    page_size: int
) -> PaginationDetails:
    num_pages = (total_items + page_size - 1) // page_size
    has_next = page < num_pages
    has_prev = page > 1
    offset = (page - 1) * page_size

    if num_pages == 0 and page == 1:
        return PaginationDetails(
            num_pages=0,
            has_next=False,
            has_prev=False,
            offset=0
        )

    if page <= 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page must be greater than 0, got {page}."
        )

    if page > num_pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page} not found. {num_pages} available."
        )

    return PaginationDetails(
        num_pages, has_next, has_prev, offset
    )


async def paginate_query[RowT: tuple[Any, ...], SerialisedT: BaseModel](
    base_statement: Select[RowT],
    session: AsyncSession,
    page: int, page_size: int,
    serialiser_func: Callable[[RowT], SerialisedT]
) -> PaginatedResults[SerialisedT]:
    count_statement = select(
        func.count(),
    ).select_from(base_statement.subquery())
    num_items = (await session.execute(count_statement)).scalar_one()
    details = get_pagination_details(num_items, page, page_size)

    if not bool(base_statement._order_by_clauses):
        logger.warning(
            f"Pagination request for {base_statement=} has no ORDER BY clause in the query.  This works, but may yield inconsistent results. Stack trace: {traceback.format_stack()}"  # noqa: E501
        )

    sized_statement = (
        base_statement
        .offset(details.offset)
        .limit(page_size)
    )

    results = await session.execute(
        sized_statement
    )

    return PaginatedResults(
        page=page,
        num_pages=details.num_pages,
        total_items=num_items,
        page_size=page_size,
        has_next=details.has_next,
        has_prev=details.has_prev,
        data=[
            serialiser_func(d._tuple())
            for d in results.all()
        ]
    )
