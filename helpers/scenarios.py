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

    # TODO: implement alternative pool/pbs scenario => only part of pool/candidate set updated at a time => need to implement all scenario dependent interfaces for new scenario as well
