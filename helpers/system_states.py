from enum import IntEnum


class SystemStates(IntEnum):
    INITIALIZATION = 0
    """
    Initiation system_state => initiation of databases, components, initial training, ...
    """

    TRAINING = 1
    """
    Active training process, components work in parallel
    """

    FINISH_TRAINING__INFO = 2
    """
    Soft end for training process => components can finish their tasks

    Order of finished jobs:
        1. AL -> empty candidate set (no more candidate evaluation incl. informativeness analysis)
        2. Oracle -> resolve all queries
        3. PL -> empty training set
    """

    FINISH_TRAINING__ORACLE = 3
    """
    Soft end for training process with AL part already done => components can finish their tasks

    Order of finished jobs:
        (1. AL -> empty candidate set (no more candidate evaluation incl. informativeness analysis)) => already finished
        2. Oracle -> resolve all queries
        3. PL -> empty training set
    """

    FINISH_TRAINING__PL = 4
    """
    Soft end for training process with AL and oracle part done => components can finish their tasks

    Order of finished jobs:
        (1. AL -> empty candidate set (no more candidate evaluation incl. informativeness analysis))  => already finished
        (2. Oracle -> resolve all queries) => already finished
        3. PL -> empty training set
    """

    TERMINATE_TRAINING = 5
    """
    Hard end of training process => components should immediately stop their tasks
    """

    PREDICT = 6
    """
    Training is finished => SL model can be used for prediction
    """

    ERROR = 7
    """
    Fatal error => model won't work
    
    - e.g., database connection can't be made
    - most likely programming error
    """
