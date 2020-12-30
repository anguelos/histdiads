import torch
from .utils import download_url, extract
import csv
from torchvision import transforms
import PIL.Image as Image


class ClammDs(torch.utils.data.Dataset):
    """Dataset class for DIA classification.

    References:
        "ICFHR2016 competition on the classification of medieval handwritings in latin script"

    """
    clamm_type_classes = ['caroline',
                          'cursiva',
                          'half_uncial',
                          'humanistic',
                          'humanistic_cursive',
                          'hybrida',
                          'praegothica',
                          'semihybrida',
                          'semitextualis',
                          'southern_textualis',
                          'textualis',
                          'uncial']

    train_url = (
    "https://clamm.irht.cnrs.fr/wp-content/uploads/ICFHR2016_CLaMM_Training.zip", 3054287145, "CLaMM_Training_Data_Set",
    "@CLaMM-filelist.csv")
    test_task1_url = (
    "https://clamm.irht.cnrs.fr/wp-content/uploads/ICFHR2016_CLaMM_task1.zip", 1509816283, "CLaMM_task1",
    "@CLaMM_task1.csv")
    test_task2_url = (
    "https://clamm.irht.cnrs.fr/wp-content/uploads/ICFHR2016_CLaMM_task2.zip", 3269954167, "CLaMM_task2",
    "@CLaMM_task2.csv")


    def _get_resources(self):
        if self.train:
            url = ClammDs.train_url
        elif not self.train and self.task == 1:
            url = ClammDs.test_task1_url
        elif not self.train and self.task == 2:
            url = ClammDs.test_task2_url
        url, filesize, subroot, csv_filename = url
        filename = url.split("/")[-1]
        filename = f"{self.root}/{filename}"
        return url, filename, filesize, subroot, csv_filename


    def load_filenames(self):
        csv_path = f"{self.root}/{self.subroot}/{self.csv_filename}"
        with open(csv_path, "r") as fin:
            csv_reader = csv.reader(open(csv_path, "r"), delimiter=';')
            self.samples = []
            name2class = dict(zip(ClammDs.clamm_type_classes, list(range(len(ClammDs.clamm_type_classes)))))
            for row in csv_reader:
                if row[1].lower() in name2class.keys():  #TODO(anguelos) handle header better
                    gt = torch.zeros(len(name2class))
                    gt[[name2class[col.lower()] for col in row[1:] if col]] = 1
                    img_path = f"{self.root}/{self.subroot}/{row[0]}"
                    self.samples.append((img_path, gt))


    def __init__(self, download=False, perform_extract=True, root="./clammds", train=True, task=1,
                 input_transform=transforms.PILToTensor(), output_transform=(lambda x: x)):
        self.download = download
        self.perform_extract = perform_extract
        self.root = root
        self.train = train
        self.task = task
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
