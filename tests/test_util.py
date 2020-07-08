"""Tests for utility functions."""
from incubator.util import chain_func, flatten2list


def test_flatten_2d() -> None:
    """Tests flatten2list for a 2D list."""
    lst = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    flat = flatten2list(lst)

    assert flat == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_flatten_3d() -> None:
    """Tests flatten2list for a 3D list."""
    lst = [
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ]
    flat = flatten2list(lst)

    assert flat == [1, 2, 3, 4, 5, 6, 7, 8]


def test_flatten_4d() -> None:
    """Tests flatten2list for a 4D list."""
    lst = [
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ]
        ],
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ]
        ]
    ]
    flat = flatten2list(lst)

    assert flat == [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]


def test_chain_func() -> None:
    """Test chain_func obtains appropriate result."""
    def multiply_2(x: int) -> int:
        return x * 2

    result = chain_func(
        1,
        multiply_2,
        multiply_2,
        multiply_2,
    )

    assert result == 8
