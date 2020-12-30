# histdiads

### Historical Document Datasets

Pytorch wrappers for historical DIA datasets.
Requires:
torchvision >= 0.4
Python 3
pytorch > 1.0

### Installation:

tested on ubuntu 18.04 but should run in any system can run pytorch.
Maybe even windows ;-)
```bash
pip3 install --user --upgrade git+https://github.com/anguelos/histdiads
```

### Usage:

```python
import histdiads
ds1 = histdiads.ManuscriptLocationDs(download=True, perform_extract=True, partition="validation")
ds2 = histdiads.ClammDs(download=True, perform_extract=True, train=False, task=2)
```

#### _download_:
Download resumes and will not download a saved file of the correct size.
Set to False, to avoid resuming, if file already downloaded or not needed.

#### _perform_extract_:
Will extract the archive file.
Set to False to avoid re-extraction if archive already extracted.

### Space requirements:

|Dataset| Size|
|-------|-----|
| ClammDs | ~19GB |
| ManuscriptLocationDs train and validation| ~35GB|

### References:
```bibtex
@inproceedings{tensmeyer2017convolutional,
  title={Convolutional neural networks for font classification},
  author={Tensmeyer, Chris and Saunders, Daniel and Martinez, Tony},
  booktitle={2017 14th IAPR international conference on document analysis and recognition (ICDAR)},
  volume={1},
  pages={985--990},
  year={2017},
  organization={IEEE}
}
```