# ActiveLearningFramework

**Bachelor Thesis:** Active Learning Strategies for Machine Learned Potentials

**Author:** Meret Unbehaun, 2022 ([EMail](mailto:meret.unbehaun@outlook.com), [LinkedIn](https://linkedin.com/in/meret-unbehaun-056467227))

Project for the **Active Learning Framework**

- parallel running PL, oracle, candidate updater, query selector
- framework runs within one cluster
- communication between components: through databases and storage for SL model

### Specifications

- utilised Python 3.10 
  - python 3 is necessary, but other versions might work as well (e.g., Python 3.7)
- Packages:
  - tqdm
  - numpy
  - dataclasses
  - if using default databases: mysql and mysql-connector-python
  
### Concrete implementations

Implement the concrete ML problem in a separate branch.
- Implement system_initiator -> will provide everything that is necessary
- Exemplary implementations on separate branches:
  - Stream-based example, house price regression: https://github.com/aimat-lab/ActiveLearningFramework/tree/example_sbs
  - Pool-based example, butene energy and force calculation: https://github.com/aimat-lab/ActiveLearningFramework/tree/Butene_energy_force_PbS
  - Stream-based example, methanol energy and force calculation: https://github.com/aimat-lab/ActiveLearningFramework/tree/Methanol_SbS

## Abbreviations:

| Abbreviation | Meaning                        |
|--------------|--------------------------------|
| AL           | Active Learner/Active Learning |
| PL           | Passive Learner                |
| ML           | Machine Learning               |
| SL           | Supervised Learning            |
| db           | database                       |
| x, y         | input, output of PL            |

