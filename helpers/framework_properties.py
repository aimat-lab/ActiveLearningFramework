waiting_times = {
    "oracle": 10,
    "pl": 10,
    "query_selector": 5
}

# SbS specific
sbs_threshold_query_decision: float = 0.5
"""
During query selection in the SbS scenario, this threshold defines whether an instance is queried or discarded:

- info(x) < thr:    x is discarded
- info(x) >= thr:   x is queried
"""

# MQS specific
mqs_threshold_query_decision: float = 0.5
"""
During query selection in the MQS scenario, this threshold defines whether an instance is queried or discarded:

- info(x) < thr:    x is discarded
- info(x) >= thr:   x is queried
"""

# PbS specific
pbs_refresh_counter_sorted_list: int = 8
"""
During query selection in PbS scenario: all instances in pool are evaluated and sorted based on their info value -> maximising instance is selected

- for efficiency: sorted list can be stored for some iterations
- from original list, the queried instance is removed and in the next iteration, the new maximising instance is selected
- list is regenerated with updated info values, after the defined refresh_counter iterations

Set value to 1, if the query selected always needs to select the maximising instance with the most recent available information
"""
