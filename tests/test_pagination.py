import pytest
import pytest_asyncio
import asyncio
from typing import Optional
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel, Field, select, create_engine
from sqlalchemy.pool import StaticPool
import tempfile
import os
from contextlib import asynccontextmanager

from pagination_utils.pagination import get_pagination_details, paginate_query, PaginationDetails
from pagination_utils.schemas import PaginatedResults


class User(SQLModel, table=True):
    __tablename__ = "test_users"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    age: int
    email: str


class UserPublic(SQLModel):
    id: int
    name: str
    age: int
    email: str


# Test fixtures
@pytest.fixture
def sample_users_data():
    """Sample user data for testing"""
    return [
        {"name": "Alice", "age": 25, "email": "alice@example.com"},
        {"name": "Bob", "age": 30, "email": "bob@example.com"},
        {"name": "Charlie", "age": 35, "email": "charlie@example.com"},
        {"name": "Diana", "age": 28, "email": "diana@example.com"},
        {"name": "Eve", "age": 32, "email": "eve@example.com"},
        {"name": "Frank", "age": 27, "email": "frank@example.com"},
        {"name": "Grace", "age": 29, "email": "grace@example.com"},
        {"name": "Henry", "age": 31, "email": "henry@example.com"},
        {"name": "Iris", "age": 26, "email": "iris@example.com"},
        {"name": "Jack", "age": 33, "email": "jack@example.com"},
    ]


@pytest_asyncio.fixture
async def async_session():
    """Create an async SQLite session for testing"""
    # Create in-memory SQLite database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Create session
    session = AsyncSession(engine)
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


@pytest_asyncio.fixture
async def populated_db(async_session, sample_users_data):
    """Populate database with test data"""
    for user_data in sample_users_data:
        user = User(**user_data)
        async_session.add(user)

    await async_session.commit()
    return async_session


class TestGetPaginationDetails:
    """Test cases for get_pagination_details function"""

    def test_first_page_with_items(self):
        """Test pagination details for first page with items"""
        details = get_pagination_details(total_items=10, page=1, page_size=3)

        assert details.num_pages == 4
        assert details.has_next == True
        assert details.has_prev == False
        assert details.offset == 0

    def test_middle_page_with_items(self):
        """Test pagination details for middle page"""
        details = get_pagination_details(total_items=10, page=2, page_size=3)

        assert details.num_pages == 4
        assert details.has_next == True
        assert details.has_prev == True
        assert details.offset == 3

    def test_last_page_with_items(self):
        """Test pagination details for last page"""
        details = get_pagination_details(total_items=10, page=4, page_size=3)

        assert details.num_pages == 4
        assert details.has_next == False
        assert details.has_prev == True
        assert details.offset == 9

    def test_single_page(self):
        """Test pagination with all items fitting in one page"""
        details = get_pagination_details(total_items=5, page=1, page_size=10)

        assert details.num_pages == 1
        assert details.has_next == False
        assert details.has_prev == False
        assert details.offset == 0

    def test_empty_results(self):
        """Test pagination with no items - should allow page 1"""
        # With 0 items, asking for page 1 should still work but return empty
        # The pagination logic should handle this gracefully
        details = get_pagination_details(total_items=0, page=1, page_size=10)

        assert details.num_pages == 0
        assert details.has_next == False
        assert details.has_prev == False
        assert details.offset == 0

    def test_exact_page_size_division(self):
        """Test when total items divide evenly by page size"""
        details = get_pagination_details(total_items=12, page=2, page_size=4)

        assert details.num_pages == 3
        assert details.has_next == True
        assert details.has_prev == True
        assert details.offset == 4

    def test_page_not_found_error(self):
        """Test HTTPException when requesting page beyond available pages"""
        with pytest.raises(HTTPException) as exc_info:
            get_pagination_details(total_items=10, page=5, page_size=3)

        assert exc_info.value.status_code == 404
        assert "Page 5 not found. 4 available." in str(exc_info.value.detail)

    def test_page_beyond_empty_results(self):
        """Test HTTPException when requesting page > 1 with empty results"""
        with pytest.raises(HTTPException) as exc_info:
            get_pagination_details(total_items=0, page=2, page_size=3)

        assert exc_info.value.status_code == 404
        assert "Page 2 not found. 0 available." in str(exc_info.value.detail)

    def test_page_zero_error(self):
        """Test HTTPException when requesting page 0"""
        with pytest.raises(HTTPException) as exc_info:
            get_pagination_details(total_items=10, page=0, page_size=3)

        assert exc_info.value.status_code == 404

    def test_negative_page_error(self):
        """Test HTTPException when requesting negative page"""
        with pytest.raises(HTTPException) as exc_info:
            get_pagination_details(total_items=10, page=-1, page_size=3)

        assert exc_info.value.status_code == 404


class TestPaginateQuery:
    """Test cases for paginate_query function"""

    @pytest.mark.asyncio
    async def test_first_page_pagination(self, populated_db):
        """Test pagination for first page"""
        base_statement = select(User)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=1,
            page_size=3,
            order_by=User.id,
            public_model=UserPublic
        )

        assert result.page == 1
        assert result.page_size == 3
        assert result.total_items == 10
        assert result.num_pages == 4
        assert result.has_next == True
        assert result.has_prev == False
        assert len(result.data) == 3
        assert result.data[0].name == "Alice"
        assert result.data[1].name == "Bob"
        assert result.data[2].name == "Charlie"

    @pytest.mark.asyncio
    async def test_middle_page_pagination(self, populated_db):
        """Test pagination for middle page"""
        base_statement = select(User)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=2,
            page_size=3,
            order_by=User.id,
            public_model=UserPublic
        )

        assert result.page == 2
        assert result.page_size == 3
        assert result.total_items == 10
        assert result.num_pages == 4
        assert result.has_next == True
        assert result.has_prev == True
        assert len(result.data) == 3
        assert result.data[0].name == "Diana"

    @pytest.mark.asyncio
    async def test_last_page_pagination(self, populated_db):
        """Test pagination for last page"""
        base_statement = select(User)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=4,
            page_size=3,
            order_by=User.id,
            public_model=UserPublic
        )

        assert result.page == 4
        assert result.page_size == 3
        assert result.total_items == 10
        assert result.num_pages == 4
        assert result.has_next == False
        assert result.has_prev == True
        assert len(result.data) == 1  # Only 1 item on last page
        assert result.data[0].name == "Jack"

    @pytest.mark.asyncio
    async def test_pagination_with_filtering(self, populated_db):
        """Test pagination with WHERE clause filtering"""
        base_statement = select(User).where(User.age >= 30)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=1,
            page_size=3,
            order_by=User.age,
            public_model=UserPublic
        )

        assert result.total_items == 5  # Users with age >= 30
        assert result.num_pages == 2
        assert len(result.data) == 3
        # Should be ordered by age: Bob(30), Henry(31), Eve(32)
        assert result.data[0].age == 30
        assert result.data[1].age == 31
        assert result.data[2].age == 32

    @pytest.mark.asyncio
    async def test_pagination_with_custom_order_by(self, populated_db):
        """Test pagination with custom ordering"""
        base_statement = select(User)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=1,
            page_size=3,
            order_by=User.age.desc(),
            public_model=UserPublic
        )

        assert len(result.data) == 3
        # Should be ordered by age descending: Charlie(35), Jack(33), Eve(32)
        assert result.data[0].age == 35
        assert result.data[1].age == 33
        assert result.data[2].age == 32

    @pytest.mark.asyncio
    async def test_pagination_no_order_by_warning(self, populated_db, caplog):
        """Test warning when no order_by is provided"""
        base_statement = select(User)

        with caplog.at_level("WARNING"):
            result = await paginate_query(
                base_statement=base_statement,
                session=populated_db,
                page=1,
                page_size=3,
                order_by=None,
                public_model=UserPublic
            )

        # Check that warning was logged
        assert "without an ORDER BY clause" in caplog.text
        assert "UserPublic" in caplog.text
        assert result.total_items == 10

    @pytest.mark.asyncio
    async def test_pagination_empty_results(self, populated_db):
        """Test pagination with no matching results"""
        base_statement = select(User).where(User.age > 100)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=1,
            page_size=3,
            order_by=User.id,
            public_model=UserPublic
        )

        assert result.total_items == 0
        assert result.num_pages == 0
        assert result.has_next == False
        assert result.has_prev == False
        assert len(result.data) == 0

    @pytest.mark.asyncio
    async def test_pagination_page_not_found(self, populated_db):
        """Test HTTPException when requesting non-existent page"""
        base_statement = select(User)

        with pytest.raises(HTTPException) as exc_info:
            await paginate_query(
                base_statement=base_statement,
                session=populated_db,
                page=10,  # Way beyond available pages
                page_size=3,
                order_by=User.id,
                public_model=UserPublic
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_pagination_large_page_size(self, populated_db):
        """Test pagination with page size larger than total items"""
        base_statement = select(User)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=1,
            page_size=20,
            order_by=User.id,
            public_model=UserPublic
        )

        assert result.page == 1
        assert result.page_size == 20
        assert result.total_items == 10
        assert result.num_pages == 1
        assert result.has_next == False
        assert result.has_prev == False
        assert len(result.data) == 10  # All items on one page

    @pytest.mark.asyncio
    async def test_pagination_maintains_existing_order_by(self, populated_db):
        """Test that existing order_by in base statement is preserved when order_by is None"""
        base_statement = select(User).order_by(User.name)

        result = await paginate_query(
            base_statement=base_statement,
            session=populated_db,
            page=1,
            page_size=3,
            order_by=None,
            public_model=UserPublic
        )

        assert len(result.data) == 3
        # Should be ordered alphabetically by name
        assert result.data[0].name == "Alice"
        assert result.data[1].name == "Bob"
        assert result.data[2].name == "Charlie"


class TestPaginationDetailsNamedTuple:
    """Test PaginationDetails NamedTuple"""

    def test_named_tuple_creation(self):
        """Test creating PaginationDetails NamedTuple"""
        details = PaginationDetails(
            num_pages=5,
            has_next=True,
            has_prev=False,
            offset=0
        )

        assert details.num_pages == 5
        assert details.has_next == True
        assert details.has_prev == False
        assert details.offset == 0

        # Test tuple unpacking
        num_pages, has_next, has_prev, offset = details
        assert num_pages == 5
        assert has_next == True
        assert has_prev == False
        assert offset == 0


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_database_error_handling(self):
        """Test handling of database errors"""
        # Create a mock session that raises an exception
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        base_statement = select(User)

        with pytest.raises(Exception) as exc_info:
            await paginate_query(
                base_statement=base_statement,
                session=mock_session,
                page=1,
                page_size=3,
                order_by=User.id,
                public_model=UserPublic
            )

        assert "Database connection failed" in str(exc_info.value)

    def test_pagination_details_with_zero_page_size(self):
        """Test behavior with zero page size"""
        with pytest.raises(ZeroDivisionError):
            get_pagination_details(total_items=10, page=1, page_size=0)


# Integration test
class TestPaginationIntegration:
    """Integration tests combining multiple aspects"""

    @pytest.mark.asyncio
    async def test_full_pagination_workflow(self, populated_db):
        """Test complete pagination workflow through multiple pages"""
        base_statement = select(User)
        all_users = []
        page = 1

        while True:
            try:
                result = await paginate_query(
                    base_statement=base_statement,
                    session=populated_db,
                    page=page,
                    page_size=3,
                    order_by=User.id,
                    public_model=UserPublic
                )
                all_users.extend(result.data)

                if not result.has_next:
                    break
                page += 1

            except HTTPException:
                break

        # Should have collected all 10 users
        assert len(all_users) == 10

        # Check that users are in correct order (by id)
        for i in range(len(all_users) - 1):
            assert all_users[i].id < all_users[i + 1].id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
