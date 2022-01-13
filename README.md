# ActiveLearningFramework

Bachelor Thesis of Meret Unbehaun, Topic: Active Learning strategies for Machine Learned potentials

Project for the **active learning framework** (including example for boston house pricing regression)

- parallel running AL/PL/Oracle
- framework runs within one cluster
- communication between components: through databases

## Project structure
Short overview over the main structure, including some descriptions.

- Scenario dependent: implies an interface implemented per scenario 
- Needs to be implemented: file/folder which containing interfaces need to be implemented per case (for every new ML problem)
```
main.py                             # controlls main workflow
al_components/
|-- candidate_update/               
    |-- candidate_updater           # scenario dependent
    |-- candidate_source            # NEEDS TO BE IMPLEMENTED (source type: scenario dependent)
|-- query_selection/                
    |-- query_selector              # scenario dependent
    |-- informativeness_analyser/   # NEEDS TO BE IMPLEMENTED
additional_component_interfaces/
|-- oracle.py                       # NEEDS TO BE IMPLEMENTED
|-- passive_learner.py              # NEEDS TO BE IMPLEMENTED
workflow_management/
|-- controller                      # controlling flow between components, parallelization
|-- database_interfaces             # communication between components, NEED TO BE IMPLEMENTED
helpers/
|-- scenario_enum.py
|-- exceptions/
```

### ML problem implementations
Implement the concrete ML problem in a separate branch.
=> tbd for boston house pricing

## Abbreviations:

| Abbreviation | Meaning                        |
|--------------|--------------------------------|
| AL           | Active Learner/Active Learning |
| PL           | Passive Learner                |
| ML           | Machine Learning               |
| SL           | Supervised Learning            |
| db           | database                       |
| x, y         | input, output of PL            |

