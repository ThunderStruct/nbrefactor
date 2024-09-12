from functools import partial

class PartialWrapper(partial):
    """
    A custom :class:`~functools.partial` subclass that supports equality.
    `__eq__` is not supported by default, and the official suggested approach
    is to sub-class `partial` (as per https://bugs.python.org/issue3564).

    This wrapper is also hashable, making it usable as a
    """

    def __eq__(self, other):
        if not isinstance(other, PartialWrapper):
            return NotImplemented

        return (self.func, self.args, self.keywords) == \
         (other.func, other.args, other.keywords)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return str((self.func, self.args, str(self.keywords) \
                     if self.keywords else None))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        # hash the function + args + keywords (as an immutable set)
        return hash((self.func, self.args, frozenset(self.keywords.items()) \
                     if self.keywords else None))


    def intersect_equals(self, other):
        if not isinstance(other, PartialWrapper):
            raise NotImplemented

        # compare func / positional arguments
        if self.func != other.func or self.args != other.args:
            return False

        # get intersection of kwargs
        intersection = set(self.keywords.keys()) & set(other.keywords.keys())

        for key in intersection:
            if self.keywords[key] != other.keywords[key]:
                return False

        return True


