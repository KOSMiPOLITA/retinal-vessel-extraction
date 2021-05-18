import os
import numpy as np
import cv2
from sys import stdout
from urllib.request import urlretrieve
from zipfile import ZipFile
from skimage.io import imread
from skimage.exposure import equalize_adapthist


class Dataset:
    @property
    def filenames(self):
        raise NotImplementedError()

    @staticmethod
    def download_progress(count, block_size, total_size):
        stdout.write("\rDownload: %.1f%%" % (float(count * block_size) / total_size * 100.0))
        stdout.flush()

    @staticmethod
    def list_directory(path):
        images_filenames = sorted(os.listdir(path))
        return [*map(lambda filename: os.path.join(path, filename), images_filenames)]

    @staticmethod
    def extract_name(path):
        return path.split('\\')[-1].split('.')[0]


class HRFDataset(Dataset):
    DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hrfdataset')
    DATASET_URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"
    IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
    MASKS_DIR = os.path.join(DATASET_DIR, 'manual1')

    def __init__(self, force_download=False):
        if not HRFDataset.__dir_exists() or force_download:
            HRFDataset.__download()

    @property
    def filenames(self):
        image_paths = Dataset.list_directory(HRFDataset.IMAGES_DIR)
        mask_paths = Dataset.list_directory(HRFDataset.MASKS_DIR)
        for image_path, mask_path in zip(image_paths, mask_paths):
            name = Dataset.extract_name(image_path)
            yield name, image_path, mask_path

    @staticmethod
    def __dir_exists():
        return os.path.exists(HRFDataset.DATASET_DIR)

    @staticmethod
    def __download():
        if not HRFDataset.__dir_exists():
            os.makedirs(HRFDataset.DATASET_DIR)
        path, _ = urlretrieve(HRFDataset.DATASET_URL, reporthook=Dataset.download_progress)
        HRFDataset.__unzip(path)
        os.remove(path)

    @staticmethod
    def __unzip(path):
        zipfile = ZipFile(path, 'r')
        zipfile.extractall(HRFDataset.DATASET_DIR)
        zipfile.close()


class DataLoader:
    def load(self, path):
        raise NotImplementedError()


class ImageLoader(DataLoader):
    def __init__(self, as_gray):
        self.__as_gray = as_gray

    def load(self, path):
        img = imread(path, as_gray=self.__as_gray)
        return img


class DataTransformation(DataLoader):
    def __init__(self, parent):
        self.__parent = parent

    def load(self, path):
        return self.transform(self.__parent.load(path))

    def transform(self, img):
        raise NotImplementedError()


class Binarize(DataTransformation):
    def transform(self, img):
        img[img > 0] = 1
        return img


class Resize(DataTransformation):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.__size = size

    def transform(self, img):
        return cv2.resize(img, self.__size)


class AddDimension(DataTransformation):
    def transform(self, img):
        return img.reshape((*img.shape, 1))


class Standarize(DataTransformation):
    def transform(self, img):
        min_value = np.amin(img)
        max_value = np.amax(img)
        if min_value == max_value:
            return img
        return (img - min_value) / (max_value - min_value)


class Clahe(DataTransformation):
    def transform(self, img):
        return equalize_adapthist(img)


class DataLoaderFactory:
    @staticmethod
    def create_data_loader(as_gray=False, binarize=False, size=None, standarize=False,
                           clahe=False, add_dimension=False):

        loader = ImageLoader(as_gray=as_gray or binarize)
        if size is not None:
            loader = Resize(loader, size=size)

        if clahe:
            loader = Clahe(loader)
        if binarize:
            loader = Binarize(loader)
        if standarize:
            loader = Standarize(loader)

        if add_dimension:
            loader = AddDimension(loader)
        return loader


class ImageDataGenerator:
    def __init__(self, paths, image_loader, mask_loader):
        self.__paths = paths
        self.__image_loader = image_loader
        self.__mask_loader = mask_loader

    def shuffle(self):
        np.random.shuffle(self.__paths)

    def __iter__(self):
        for name, image_path, mask_path in self.__paths:
            yield self.__image_loader.load(image_path), self.__mask_loader.load(mask_path)

    @property
    def size(self):
        return len(self.__paths)


class DatasetLoader:
    def __init__(self, dataset, seed=None, max_size=None, validation_size=5):
        random_state = np.random.RandomState(seed=seed)
        paths = random_state.permutation([*dataset.filenames])
        if max_size is not None:
            paths = paths[:max_size]
        self.__validation_paths = paths[:validation_size]
        self.__train_paths = paths[validation_size:]

    def load_training(self, *args, **kwargs):
        return DatasetLoader.__create_generator(self.__train_paths, *args, **kwargs)

    def load_validation(self, raw_image_loader, *args, **kwargs):
        generator = DatasetLoader.__create_generator(self.__validation_paths, *args, **kwargs)
        for (name, image_path, _), (image, mask) in zip(self.__validation_paths, generator):
            yield name, raw_image_loader.load(image_path), image, mask

    @staticmethod
    def __create_generator(paths, image_loader, mask_loader):
        generator = ImageDataGenerator(paths, image_loader, mask_loader)
        return generator
