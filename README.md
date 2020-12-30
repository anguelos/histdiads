# histdiads

### Historical Document Datasets

Pytorch wrappers for historical DIA datasets.
Requires:
torchvision >= 0.4,
Python >= 3.6,
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
This is the space required both for the archives and the extracted data.
Once the data are extracted, the archives can be erased but _*download*_ 
and _perform_extract_ should be set to False if so.

### References:
```bibtex
@inproceedings{cloppet2016icfhr2016,
  title={ICFHR2016 competition on the classification of medieval handwritings in latin script},
  author={Cloppet, Florence and Eglin, V{\'e}ronique and Stutzmann, Dominique and Vincent, Nicole and others},
  booktitle={2016 15th International Conference on Frontiers in Handwriting Recognition (ICFHR)},
  pages={590--595},
  year={2016},
  organization={IEEE}
}
```