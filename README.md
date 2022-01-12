# ActiveLearningFramework

Bachelor Thesis of Meret Unbehaun, Topic: Active Learning strategies for Machine Learned potentials

Project for the **active learning framework** (including example for boston house pricing regression)

- parallel running AL/PL/Oracle
- framework runs within one cluster
- communication between components: through databases

## Project structure

```
main.py                 # controlls main workflow
al_components/
|-- candidate_update/   # scenario dependent
|-- query_selection/    # scenario dependent
    |-- informativeness_analyser/  # needs to be implemented
additional_component_interfaces/
|-- oracle.py           # needs to be implemented
|-- passive_learner.py  # needs to be implemented
workflow_management/
|-- controller          # controlling flow between components, parallelization
|-- database_interfaces # need to be implemented => communication between components
helpers/
|-- scenario_enum.py
|-- exceptions/
```

## Abbreviations:

| Abbreviation | Meaning                           |
|--------------|-----------------------------------|
| AL           | Active Learner/Active Learning    |
| PL           | Passive Learner                   |
| ML           | Machine Learning                  |
| db           | database                          |
| x, y         | input, output of passive learner  |

