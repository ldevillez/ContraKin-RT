# ContraKin-RT

[![DOI](https://zenodo.org/badge/1149863443.svg)](https://doi.org/10.5281/zenodo.18509311)

This repository contains the code for the paper "Design and Control of an Active Prosthesis for Replacing the Hip after Disarticulation".

The code estimates the kinematics of a hip joint based on the kinematics of the contralateral leg using:
- **CLME (Complementary Limb Motion Estimation)** ([paper](https://ieeexplore.ieee.org/abstract/document/4668434))
- **AO (Adaptive Oscillators)** ([paper](https://ieeexplore.ieee.org/abstract/document/6428719))

## Folder Structure
- `datasets/`: Contains the datasets used for evaluation.
- `src/`: Contains the source code for the project.

## Datasets
The datasets used for evaluation are stored in the `datasets/` folder. The code is made to interface directly with the Devillez dataset.

To load and use datasets, refer to the [dataset_manager.py](src/dataset_manager.py) script.

## Requirements
Install dependencies with:
```
pip install -r requirements.txt
```

## Main Code Blocks

### Example Script
- [`example.py`](src/example.py): Demonstrates usage of the main estimators, including loading data, running estimations, and plotting results.

### Estimators
- [`ao_estimator.py`](src/ao_estimator.py): Implements the Adaptive Oscillators estimator for hip kinematics.
- [`clme_estimator.py`](src/clme_estimator.py): Implements the Complementary Limb Motion Estimator.
- [`cao_estimator.py`](src/cao_estimator.py): Combines CLME and AO estimators for smooth transitions.
- [`standing_estimator.py`](src/standing_estimator.py): Estimates standing phases using exponential filtering.
- [`standwalk_estimator.py`](src/standwalk_estimator.py): Combines standing and walking estimators for transitions.

### Data Management
- [`dataset_manager.py`](src/dataset_manager.py): Functions to list datasets, subjects, trials, and load data.
- [`data_manager.py`](src/data_manager.py): Classes for loading, accessing, and plotting data; supports iteration over multiple datasets/trials.

### Utilities
- [`tasks.py`](src/tasks.py): Division of the trials into tasks for post processing
- [`parameter_manager.py`](src/parameter_manager.py): Manages estimator parameters.
- ['cycler_decomposition.py'](src/cycler_decomposition.py): Implement cycle by cycle decompositon of the walking task.

## Usage

See [`example.py`](src/example.py) for a demonstration of loading data and running estimators:

```python
from data_manager import DataManager
from ao_estimator import AO

dm = DataManager()
dm.load_data("P01", "exp_transition_03-S3_01")
ao = AO(dm)
ao.compute()
fig, axs = ao.complete_plot()
```

## Real-Time Implementation
No real-time implementation is provided in this repository as it will depend on your specific hardware and software environment. However, the estimators are designed to work in real-time applications. You can adapt the code from the estimators to your real-time system. A transposition was done to work with ROS2.

## License
This project is licensed under the GNU Lesser General Public License - see the [LICENSE](LICENSE) file for details.
