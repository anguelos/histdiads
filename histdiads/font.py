import torch
from torchvision import transforms
from pathlib import Path
from .utils import download_url, extract
import csv
from PIL import Image

class FontDs(torch.utils.data.Dataset):
    """Dataset class for DIA classification.

    Classes are font types of historical printed texts.

    """
    font_types = {'antiqua':0,
                  'bastarda':1,
                  'fraktur':2,
                  'greek':3,
                  'hebrew':4,
                  'italic':5,
                  'rotunda':6,
                  'schwabacher':7,
                  'textura':8}

    train_validation_urls = [('https://zenodo.org/record/3366686/files/fontgroupsdataset-a.zip?download=1,', 6271470206),
                             ('https://zenodo.org/record/3366686/files/fontgroupsdataset-b.zip?download=1', 6205033403),
                             ('https://zenodo.org/record/3366686/files/fontgroupsdataset-c.zip?download=1', 6255105006),
                             ('https://zenodo.org/record/3366686/files/fontgroupsdataset-d.zip?download=1', 6363950353),
                             ('https://zenodo.org/record/3366686/files/fontgroupsdataset-e.zip?download=1', 6268677152),
                             ('https://zenodo.org/record/3366686/files/fontgroupsdataset-f.zip?download=1', 6549675939),
                             ('https://zenodo.org/record/3366686/files/fontgroupsdataset-g.zip?download=1', 6298730268),
                             ('https://zenodo.org/record/3366686/files/fontgroupsdataset-labels.zip?download=1', 291986)]
    test_urls = [("NA", -1)]

    def __init__(self, download=False, perform_extract=True, root=".", partition="train",
                 input_transform=transforms.PILToTensor(), output_transform=(lambda x: x)):
        """

        :param download: Whether to download required archives, if archive is partially there, the download resumes.
        :param perform_extract: Whether to extract the downloaded archives.
        :param root: folder in which the archives will be downloaded and extracted.
        :param partition: one of ["train", "validation", "test"]. Using train or validation requires > 50GB of storage.
        :param input_transform:
        :param output_transform:
        """
        assert partition in ["train", "validation", "test"]
        self.download = download
        self.perform_extract = perform_extract
        self.root = Path(root)
        self.partition = partition

        self.input_transform = input_transform
        self.output_transform = output_transform
        urls, zip_filenames, filesizes, self.subroot, self.csv_filename = self._get_resources()

        if self.download:
            for n in range(len(urls)):
                download_url(urls[n], zip_filenames[n], filesizes[n])

        if self.perform_extract:
            for zip_filename in zip_filenames:
                extract(zip_filename, self.root)
        self.load_filenames()

    def _get_resources(self):
        if self.partition == "train":
            urls = FontDs.train_validation_urls
            csv_filename = Path.joinpath(self.root, "labels-training.csv")
        elif self.partition == "validation":
            urls = FontDs.train_validation_urls
            csv_filename = Path.joinpath(self.root, "labels-test.csv")
        elif self.partition == "test":
            urls = FontDs.test_urls
            csv_filename = Path.joinpath(self.root, "TODO.csv")
        else:
            raise ValueError
        filesizes = [url[1] for url in urls]
        urls = [url[0] for url in urls]

        # url, filesize, subroot, csv_filename = url
        filenames = [url.split("/")[-1].split("?")[0] for url in urls]
        # filename = f"{self.root}/{filename}"
        filenames = [Path.joinpath(self.root, filename) for filename in filenames]
        return urls, filenames, filesizes, "./", csv_filename

    def load_filenames(self):
        csv_path = Path.joinpath(self.root, self.subroot, self.csv_filename)
        with open(csv_path, "r") as fin:
            lines = fin.read().strip().split("\n")
            valid_fonts = FontDs.font_types.keys()
            lines = [line.split(",") for line in lines]
            lines = [line for line in lines if len(line)==2 and line[1] in valid_fonts] # keeping single class lines
            self.samples = [(Path.joinpath(self.root, self.subroot, line[0].strip()), FontDs.font_types[line[1].strip()]) for line in lines]

    def __getitem__(self, item):
        input_path, gt = self.samples[item]
        input_img = self.input_transform(Image.open(input_path))
        gt = self.output_transform(gt)
        return input_img, gt

    def __len__(self):
        return len(self.samples)
