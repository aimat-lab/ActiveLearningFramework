from typing import TypeVar

X = TypeVar("X")
"""
Type of the input of instances

- most likely numpy array
- input of the function taught to SL model:   X -> Y
"""

Y = TypeVar("Y")
"""
Type of the output of labelled instances

- output of the function taught to SL model:   X -> Y
"""

AddInfo_Y = TypeVar("AddInfo_Y")
"""
Additional information about the prediction the current PL can perform => e.g. (uncertainty, )
"""

CandInfo = TypeVar("CandInfo")
"""
Type for the additional information added to candidates

- often used: predictions (ensure has type Y) and uncertainty
"""
