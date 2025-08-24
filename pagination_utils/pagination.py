from typing import Annotated, Any, NamedTuple

from fastapi import Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, func, select
from sqlmodel.sql.expression import SelectOfScalar

import logging
from schemas.pagination import PaginatedResults


logger = logging.getLogger(__name__)



class PaginationDetails(NamedTuple):
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

    if page > num_pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page} not found. {num_pages} available."
        )

    return PaginationDetails(
        num_pages, has_next, has_prev, offset
    )


async def paginate_query[Model: SQLModel, PublicModel: SQLModel](
    base_statement: SelectOfScalar[Model],
    session: AsyncSession,
    page: int, page_size: int,
    order_by: Any | None,
    public_model: type[PublicModel]
) -> PaginatedResults[PublicModel]:
    count_statement = select(
        func.count(),
    ).select_from(base_statement.subquery())
    num_items = (await session.execute(count_statement)).scalar_one()
    details = get_pagination_details(num_items, page, page_size)

    if order_by is None:
        ordered_statement = base_statement
    else:
        ordered_statement = base_statement.order_by(order_by)

    if not bool(ordered_statement._order_by_clauses):
        logger.warn(
            f"Pagination statement for {public_model} without an ORDER BY"
            " clause in the query. This works, but may yield inconsistent"
            " results."
        )

    sized_statement = (
        ordered_statement
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
            public_model.model_validate(d)
            for d in results.scalars()
        ]
    )
