# foReal - Team up for real-world data exploration

foReal offers data processing for real-world sensor deployments. A multi-modal, multi-application data processing and analysis framework designed for data exploration.

Please refer to the documentation for a full user guide

**NOTE: This framework is experimental and still under development. It's by no means production-ready and might break anytime!**

## Why foReal?
Open environmental data can be used to tackle environmental challenges. However, such data is often incomprehensible since it is inherently complex and thus limited to a tech-savvy group with environmental background. Improved accessibility can lead to better collaboration between experts and decision-makers, involvement of the public and machine intelligence and consequently a society-wide response to pending challenges.

foReal follows a declarative paradigm: To bridge expertise domains it is more relevant to define what is needed (declarative), instead of how it is implemented (imperative). To accomplish a declarative workflow, foReal focuses on

- Configurability: Define your analysis workflow using a task graph which can be dynamically reconfigured by you or your project partners.
- Accessbility: Integrated web service to show, share and debug your analaysis.
- Reproducability: Define your task graph in a python file and load it by different apps to perform the same computations.

## Installation

Download and install [miniconda for python](https://docs.conda.io/en/latest/miniconda.html)

```
conda create -n foreal -y python=3.8
conda activate foreal
pip install poetry
pip install numba
poetry install
```


## Configuration
Some foReal function depend on specific directories or require specific access keys. These keys are stored in a configuration file. The location of the file is OS-dependant. An empty file can be created with (its location will be printed):

```
python -c "import foreal; foreal.create_user_config_file()"
```


You can open the file and insert configuration parameters in `yaml` format e.g.

```
user_dir: /path/to/usr/dir
arclink: {"user":"theusername", "password":"topsecretpw", "url":"http://arclink.server.address"}
```

## Documentation
The documentation is created with Markdown and pydoc-markdown. From within the `docs` folder, the documentation can compiled by invoking `novella` or it can be served using `novella --serve`.

## Caveats
* This public release was copied from an internal version of the framework to support open-sourcing our research projects. Not all parts (certain features, tests, benchmarks, ...) have been transfered yet.
* Currently some functions are designed to work with time-series data only, meaning the data must have a `time` coordinate. This will be resolved in the future.

<!-- Links -->
[foreal]: https://www.foreal.io


