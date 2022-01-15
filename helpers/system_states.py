from enum import IntEnum


class SystemStates(IntEnum):
    Init = 0
    """
    Initiation system_state => initiation of databases, components, initial training, ...
    """

    Training = 1
    """
    Active training process, components work in parallel
    """

    FinishTraining = 2
    """
    Soft end for training process => components can finish their tasks
    
    Order of finished jobs:
        1. AL -> no more informativeness analysis, no new queried
        2. candidate update -> no more candidates need to be evaluated
        3. Oracle -> finish all queries
        4. PL -> no more new training data will come up 
    """

    TerminateTraining = 3
    """
    Hard end of training process => components should immediately stop their tasks
    """

    Predict = 4
    """
    Training is finished => SL model can be used for prediction
    """

    Error = 5
    """
    Fatal error => model won't work
    
    - e.g., database connection can't be made
    - most likely programming error
    """
