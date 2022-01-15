import enum


class Scenarios(enum.Enum):
    PbS = 1
    """
    Pool based sampling scenario
    """

    SbS = 2
    """
    Stream based sampling scenario
    """

    MQS = 3
    """
    Membership query synthesis => query generation
    """
