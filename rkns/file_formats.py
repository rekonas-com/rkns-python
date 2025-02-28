from enum import Enum


class FileFormat(Enum):
    """


    Parameters
    ----------
    Enum
        _description_
    """

    RKNS = 0
    EDF = 1
    EDF_PLUS = 2
    BDF = 3
    BDF_PLUS = 4

    UNKNOWN = -1
