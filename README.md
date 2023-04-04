# TDG Project

## Development

Clone the repository, create an conda environment from file

`conda env create -f environment.yml`

Choose the environment for the project in your IDE.

Then, pip install the project for development

`pip install -e .`

### Data

The data of the images should be in such hierarchy 

data
| training
  | label1
  | label2
| testing
  | label1
  | label2

Put all the data in training folder, run `testing_transfer.py` to randomise the testing images