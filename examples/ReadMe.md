# foReal Examples

After installing the framework we can run two examples for one set of environmental data. The examples explain how to run the integrated webportal and how to create a dataset for machine learning.
Currently all explanations about how to use foreal are embedded as comments in the respective files.

To run the examples go to the example directory, e.g. `cd examples/permalytix`

## Webportal
The relevant files for the webportal are [webportal.py](https://github.com/niowniow/foreal/blob/main/examples/permalytix/src/webportal.py) and [simple.py](https://github.com/niowniow/foreal/blob/main/examples/permalytix/src/taskgraphs/simple.py). The following command sets up and runs the webportal

```
python -m src.webportal
```

## Machine Learning

The example trains a simple pytorch neural network. We first need to install additional packages
```
conda install -y scikit-learn pytorch torchvision torchaudio cudatoolkit=11.3 ignite numpy==1.22 -c pytorch
```

The relevant files are [train.py](https://github.com/niowniow/foreal/blob/main/examples/permalytix/src/train.py) and [classification.py](https://github.com/niowniow/foreal/blob/main/examples/permalytix/src/taskgraphs/classification.py)

Then we can run the example
```
python -m src.train
```
