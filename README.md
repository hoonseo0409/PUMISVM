Experiments codes with synthetic multi instance data.

- Project Structure:
run.py: Entry file to run the project.
utils.py: Utility functions for experiments.
PUMISVM.py: PUMISVM model.
other_models.py: Baseline Positive and Unlabeled (PU) learning or outlier detection models for comparison.

- Instructions to conduct classification on synthetic dataset.
1. Install the packages with command "pip install requirements.py"
2. Run run.py with "python3 run.py"

- Dependencies
Tested on Python 3.9.16 and MacOS Sonoma 14.4.


# Positive and Unlabeled Multi-Instance Support Vector Machine (PUMISVM)
This repository contains a collection of Python scripts implementing Positive and Unlabeled Multi-Instance Support Vector Machine (PUMISVM) model and its experiments on synthetic PU multi-instance dataset. It creates the data creation, building baseline models, calculating classification performance metrics, and visualizing the results.

## Getting Started
### Dependencies

- Python 3.9.16
- Python packages listed in requirements.txt

### Installation
To set up the project, start by cloning the repository to your local machine. We highly recommend you to create the isolated Python environment via conda, for example:
```
conda create -n "PUMISVM" python=3.9
```

Then you can activate the created environment:
```
conda activate "PUMISVM"
```

Then you can install the required packages by:
```
pip install requirements.py
```

## Running the tests

After installing the packages you can conduct the experiments by:

```
python3 run.py
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
