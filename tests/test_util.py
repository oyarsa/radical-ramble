from incubator.util import chain_func, flatten2list

def test_flatten_2d() -> None:
    lst = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    flat = flatten2list(lst)

    assert flat == [1,2,3,4,5,6,7,8,9]

def test_flatten_3d() -> None:
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
