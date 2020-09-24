import typing as t
import enum


def _identity(x):
    return x


class SearchStop(enum.Enum):
    GTEQ = 0
    GT = 1


def search(sorted_sequence: t.Union[list, tuple], val, key=None, search_stop: SearchStop = SearchStop.GTEQ):
    """Get the index of the first element >= or > val using binary search

    :param sorted_sequence: a sorted sequence with each element comparable to val
    :param key: if not None, a function of one argument to extract a comparison key from each element
    :param search_stop: decide whether to stop on first element >= (GTEQ) or > (GT) val
    :return: index of first element >= or > val, otherwise len(sorted_sequence)
    """

    key = key or _identity
    length = len(sorted_sequence)

    left = 0
    right = length - 1

    val = key(val)

    while left <= right:
        m = (left + right) // 2

        m_val = key(sorted_sequence[m])

        if m_val < val:
            left = m + 1
        elif val < m_val:
            right = m - 1
        else:
            left = m
            break

    # all elements bigger than val
    if right < 0:
        return 0

    if search_stop is SearchStop.GTEQ:
        while left > 0 and key(sorted_sequence[left - 1]) == val:
            left -= 1
    elif search_stop is SearchStop.GT:
        while left < length and key(sorted_sequence[left]) == val:
            left += 1
    else:
        raise RuntimeError("Unknown search stop {}".format(search_stop))

    return left
