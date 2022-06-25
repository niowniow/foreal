# foReal Examples

After installing the framework we can run two examples for one set of environmental data. The examples explain how to run the integrated webport and how to create a dataset for machine learning.
Currently all explanations about how to use foreal are embedded as comments in the respective files

 - Webportal: 
 - 


To run the examples go to the example directory, e.g. `cd examples/permalytix`

## Webportal
The relevant files for the webportal are [webportal.py](../../examples/permalytix/src/webportal.py) and [simple.py](../../examples/permalytics/src/taskgraphs/simple.py). The following command sets up and runs the webportal

```
python -m src.webportal
```

## Machine Learning

The example trains a simple pytorch neural network. We first need to install additional packages
```
conda install -y scikit-learn pytorch torchvision torchaudio cudatoolkit=11.3 ignite numpy==1.22 -c pytorch
```

The relevant files are [train.py](../../examples/permalytix/src/train.py) and [classification.py](../../examples/permalytics/taskgraphs/src/classification.py)

Then we can run the example
```
python -m src.train
```
