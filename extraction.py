import cv2
import numpy as np
from scipy.stats import moment


class PointGenerator:
    @property
    def points(self):
        raise NotImplementedError()


class PointGeneratorAll(PointGenerator):
    def __init__(self, shape):
        self.__shape = shape

    @property
    def points(self):
        for y in range(self.__shape[0]):
            for x in range(self.__shape[1]):
                yield y, x


class PointGeneratorRandom(PointGenerator):
    def __init__(self, shape, count, seed=None):
        self.__shape = shape
        self.__count = count
        self.__seed = seed

    @property
    def points(self):
        random_state = np.random.RandomState(seed=self.__seed)
        return zip(
            random_state.randint(self.__shape[0], size=self.__count),
            random_state.randint(self.__shape[1], size=self.__count),
        )


class PointGeneratorBalanced(PointGenerator):
    def __init__(self, mask, count, seed=None, balance=0.5):
        self.__count = count
        positive = np.where(mask == 1)
        negative = np.where(mask == 0)
        random_state1 = np.random.RandomState(seed=seed)
        random_state2 = np.random.RandomState(seed=seed)
        random_state1.shuffle(positive[0])
        random_state1.shuffle(negative[0])
        random_state2.shuffle(positive[1])
        random_state2.shuffle(negative[1])
        count = min(count, mask.shape[0] * mask.shape[1])
        pos_count = min(int(count * balance), len(positive[0]))
        neg_count = count - pos_count
        self.__y = np.concatenate((positive[0][:pos_count], negative[0][:neg_count]))
        self.__x = np.concatenate((positive[1][:pos_count], negative[1][:neg_count]))
        random_state1.shuffle(self.__y)
        random_state2.shuffle(self.__x)

    @property
    def points(self):
        return zip(self.__y, self.__x)


class Extractor:
    def extract(self):
        raise NotImplementedError()


class PatchExtractor(Extractor):
    def __init__(self, image, patch_size, point_generator):
        super().__init__()
        self.__patch_size = patch_size
        self.__center = (patch_size // 2, patch_size // 2)
        self.__point_generator = point_generator
        self.__image = PatchExtractor.padded_image(image, patch_size)

    def extract(self):
        for y, x in self.__point_generator.points:
            yield self.__patch_at(y, x)

    def __patch_at(self, y, x):
        return self.__image[y:(y + self.__patch_size), x:(x + self.__patch_size)]

    @staticmethod
    def padded_image(image, patch_size):
        l_padding = patch_size // 2
        r_padding = l_padding - 1 + (patch_size % 2)
        height, width = image.shape[:2]
        padded_image = np.zeros((height + patch_size - 1, width + patch_size - 1, *image.shape[2:]))
        padded_image[l_padding:-r_padding, l_padding:-r_padding] = image
        return padded_image

    @staticmethod
    def extract_from_padding(padded_image, patch_size):
        l_padding = patch_size // 2
        r_padding = l_padding - 1 + (patch_size % 2)
        return padded_image[l_padding:-r_padding, l_padding:-r_padding]


class FeatureExtractor(Extractor):
    def __init__(self, image, patch_size, point_generator):
        super().__init__()
        self.__center = (patch_size // 2, patch_size // 2)
        self.__patch_extractor = PatchExtractor(image, patch_size, point_generator)

    def extract(self):
        for patch in self.__patch_extractor.extract():
            yield self.__patch_to_features(patch)

    def __patch_to_features(self, patch):
        return [
            patch[self.__center],
            np.mean(patch),
            np.std(patch),
            moment(patch, moment=3, axis=None)
        ]

    @staticmethod
    def __hu_moments(patch):
        th = patch.copy()
        th[th <= 0.5] = 0
        th[th > 0.5] = 1
        return cv2.HuMoments(cv2.moments(th)).flatten()[:6]


class PixelExtractor(Extractor):
    def __init__(self, image, point_generator):
        self.__image = image
        self.__point_generator = point_generator

    def extract(self):
        for y, x in self.__point_generator.points:
            yield self.__image[y, x]
