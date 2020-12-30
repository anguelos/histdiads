from .utils import download_url, extract
import torch
import csv
from torchvision import transforms
from PIL import Image
from pathlib import Path

class ManuscriptLocationDs(torch.utils.data.Dataset):
    """Dataset class for DIA classification.

    Classes are locations of manuscripts.

    """
    locations = ["Cluny", "Corbie", "Citeaux",
                 "Florence", "Fonteney", "Himanis",
                 "Milan", "MontSaintMichel", "Paris",
                 "SaintBertin", "SaintGermainDesPres",
                 "SainMatrialDeLinoges", "Signy"]

    train_url = (
    "http://158.109.8.13/assets/historical_location/location_dia_ds_v2_train.tar.gz", 18488630441, "./",
    "location_dia_ds/train_only_samples/groundtruth.csv")
    validation_url = (
    "http://158.109.8.13/assets/historical_location/location_dia_ds_v2_validation.tar.gz", 182585320, "./",
    "location_dia_ds/validation/groundtruth.csv")
    test_url = (
    "http://158.109.8.13/assets/historical_location/location_dia_ds_v2_test.tar.gz", 905861487, "./",
    "location_dia_ds/test/groundtruth.csv")

    def _get_resources(self):
        if self.partition == "train":
            url = ManuscriptLocationDs.train_url
        elif self.partition == "validation":
            url = ManuscriptLocationDs.validation_url
            print("URL:", url)
        elif self.partition == "test":
            url = ManuscriptLocationDs.test_url
        else:
            raise ValueError
        url, filesize, subroot, csv_filename = url
        filename = url.split("/")[-1]
        #filename = f"{self.root}/{filename}"
        filename = Path.joinpath(self.root, filename)
        return url, filename, filesize, subroot, csv_filename


    def load_filenames(self):
        #csv_path = f"{self.root}/{self.subroot}/{self.csv_filename}"
        csv_path = Path.joinpath(self.root, self.subroot, self.csv_filename)
        with open(csv_path, "r") as fin:
            csv_reader = csv.reader(open(csv_path, "r"), delimiter=',')
            self.samples = []
            name2class = dict(zip(ManuscriptLocationDs.locations, list(range(len(ManuscriptLocationDs.locations)))))
            for row in csv_reader:
                if row[1] in ManuscriptLocationDs.locations:
                    gt = torch.zeros(len(name2class))
                    gt[[name2class[col] for col in row[1:]]] = 1
                    #img_path = f"{self.root}/{self.subroot}/{row[0]}"
                    img_path = Path.joinpath(self.root, self.subroot, row[0])
                    self.samples.append((img_path, gt))


    def __init__(self, download=False, perform_extract=True, root=".", partition="train",
                 input_transform=transforms.PILToTensor(), output_transform=(lambda x: x)):
        assert partition in ["train", "validation", "test"]
        self.download = download
        self.perform_extract = perform_extract
        self.root = Path(root)
        self.partition = partition

        self.input_transform = input_transform
        self.output_transform = output_transform
        url, zip_filename, filesize, self.subroot, self.csv_filename = self._get_resources()

        if self.download:
            download_url(url, zip_filename, filesize)

        if self.perform_extract:
            extract(zip_filename, self.root)
        self.load_filenames()

    def __getitem__(self, item):
        img_path, gt = self.samples[item]
        img = Image.open(img_path)
        return self.input_transform(img), self.output_transform(gt)

    def __len__(self):
        return len(self.samples)
