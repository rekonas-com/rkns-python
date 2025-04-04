class RKNSParseError(Exception):
    pass


class GroupComparisonError(Exception):
    """Base exception for group comparison failures."""

    pass


class NameMismatchError(GroupComparisonError):
    """Raised when group names do not match."""

    pass


class MemberCountMismatchError(GroupComparisonError):
    """Raised when the number of members in groups do not match."""

    pass


class PathMismatchError(GroupComparisonError):
    """Raised when keys (paths) of members do not match."""

    pass


class ArrayShapeMismatchError(GroupComparisonError):
    """Raised when array shapes do not match."""

    pass


class ArrayValueMismatchError(GroupComparisonError):
    """Raised when array values do not match."""

    pass


class GroupPathMismatchError(GroupComparisonError):
    """Raised when nested group names do not match."""

    pass


class AttributeMismatchError(GroupComparisonError):
    """Raised when attributes do not match."""

    pass
