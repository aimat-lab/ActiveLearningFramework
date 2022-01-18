from typing import TypeVar

X = TypeVar("X")
"""
Type of the input of instances

- most likely a numpy array
- typing of equality trying to be taught to SL model:   X -> Y
"""

# TODO: will type of the predictions always align with actual labels? => if not introduce new TypeVar
Y = TypeVar("Y")
"""
Type of the output of labelled instances

- typing of equality trying to be taught to SL model:   X -> Y
"""

AddInfo_Y = TypeVar("AddInfo_Y")
"""
Additional information about the prediction the current PL can perform => e.g. uncertainty
"""

CandInfo = TypeVar("CandInfo")
"""
Type for the additional information added to candidates

- often used: predictions (ensure has type Y) and uncertainty
"""