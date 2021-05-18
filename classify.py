import time
import os
import joblib
from sys import stdout
from sklearn.neighbors import KNeighborsClassifier
from skimage.filters import threshold_otsu, unsharp_mask
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.metrics import Precision, Recall
from keras.optimizers import Adam

import image_processingv4
from image_processing import *
from extraction import PointGeneratorAll, PointGeneratorBalanced, PatchExtractor, \
    FeatureExtractor, PixelExtractor


class ClassificationResult:
    LABELS = {
        'name': 'Name',
        'duration': 'Duration',
        'tp': 'True positive',
        'fp': 'False positive',
        'fn': 'False negative',
        'tn': 'True negative',
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity'
    }
    DEFAULT_FORMAT = "%s"
    PERCENT_FORMAT = "%.1f%%"
    FORMATS = {
        'duration': "%.1f s",
        'accuracy': PERCENT_FORMAT,
        'sensitivity': PERCENT_FORMAT,
        'specificity': PERCENT_FORMAT
    }

    def __init__(self, name, input_image, expected_result, actual_result, duration):
        self.__input_image = input_image
        self.__expected_result = expected_result
        self.__actual_result = actual_result

        self.__data = dict()
        self.__add_data('name', name)
        self.__add_data('duration', duration)

        self.__calculate_statistics()

    def __getitem__(self, item):
        return self.__data[item]

    @property
    def input_image(self):
        return self.__input_image

    @property
    def expected_result(self):
        return self.__expected_result

    @property
    def actual_result(self):
        return self.__actual_result

    @property
    def error_matrix(self):
        return self.__error_matrix

    @property
    def data(self):
        for k, v in self.__data.items():
            yield ClassificationResult.get_label(k), \
                  ClassificationResult.get_format(k) % v

    def __add_data(self, key, value):
        self.__data[key] = value

    def __calculate_statistics(self):
        equal = self.__expected_result == self.__actual_result
        notequal = self.__expected_result != self.__actual_result
        positive = self.__actual_result == 1
        negative = self.__actual_result == 0

        self.__generate_error_matrix(equal, notequal, positive, negative)
        self.__calculate_confusion_matrix(equal, notequal, positive, negative)
        self.__calculate_accuracy()
        self.__calculate_sensitivity()
        self.__calculate_specificity()

    def __generate_error_matrix(self, equal, notequal, positive, negative):
        matrix = np.zeros((*self.__actual_result.shape, 3))
        matrix[equal & positive, :] = 1
        matrix[notequal & positive, 2] = 1
        matrix[notequal & negative, 0] = 1
        self.__error_matrix = matrix

    def __calculate_confusion_matrix(self, equal, notequal, positive, negative):
        self.__add_data('tp', (equal & positive).sum())
        self.__add_data('fp', (notequal & positive).sum())
        self.__add_data('fn', (notequal & negative).sum())
        self.__add_data('tn', (equal & negative).sum())

    def __calculate_accuracy(self):
        tp = self.__data['tp']
        fp = self.__data['fp']
        fn = self.__data['fn']
        tn = self.__data['tn']
        self.__add_data('accuracy', 100.0 * (tp + tn) / (tp + tn + fp + fn))

    def __calculate_sensitivity(self):
        tp = self.__data['tp']
        fn = self.__data['fn']
        self.__add_data('sensitivity', 100.0 * tp / (tp + fn))

    def __calculate_specificity(self):
        fp = self.__data['fp']
        tn = self.__data['tn']
        self.__add_data('specificity', 100.0 * tn / (fp + tn))

    @staticmethod
    def get_label(key):
        return ClassificationResult.LABELS[key]

    @staticmethod
    def get_format(key):
        return ClassificationResult.FORMATS.get(key, ClassificationResult.DEFAULT_FORMAT)


class Classifier:
    NAME = None
    MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')

    def train(self, train_generator):
        raise NotImplementedError()

    def classify(self, name, raw_image, input_image, expected_result):
        stdout.write("Classify...")
        start_time = time.time()
        actual_result = self._inner_classify(input_image)
        duration = time.time() - start_time
        stdout.write("\rClassification done\n")
        return ClassificationResult(
            name, raw_image, expected_result.reshape(expected_result.shape[:2]),
            actual_result, duration)

    def _inner_classify(self, input_image):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    @staticmethod
    def create_model_directory():
        if not os.path.exists(Classifier.MODEL_DIR):
            os.mkdir(Classifier.MODEL_DIR)


class ImageProcessingClassifier(Classifier):
    NAME = "Image Processing"

    def train(self, train_generator):
        pass

    def _inner_classify(self, input_image):
        image = input_image.copy()
        channels = rgb_split(image)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = cv2.cvtColor(channels[1], cv2.COLOR_BGR2GRAY)
        image = clahe.apply(image)
        image = normalize(image, 0.28, 98)

        image = clear_data(image)
        image = correcting(image)

        image = put_mask(image, create_mask(channels[0]))
        return image

    def save(self):
        pass

    def load(self):
        pass


class ImageProcessingClassifierTurboExtra(ImageProcessingClassifier):
    NAME = "Image Processing v2"

    def _inner_classify(self, input_image):
        image = input_image.copy()
        channels = rgb_split(image)

        mask = create_mask(channels[0])

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_green = cv2.cvtColor(put_mask(channels[1], mask), cv2.COLOR_BGR2GRAY)
        img_clahe = clahe.apply(img_green)

        veils = find_veils(img_clahe)
        veils = put_mask(veils, mask)

        veils_fan = normalize(veils.copy(), 0.28, 98)
        veils_fan = contrast(veils_fan, 0.25)
        return veils_fan


class ImageProcessingClassifierV4(ImageProcessingClassifier):
    NAME = "Image Processing v4"

    def _inner_classify(self, input_image):
        image = input_image

        # RGB SPLIT
        channels = image_processingv4.rgb_split(image)

        # MASK CREATION
        mask = image_processingv4.create_mask(channels[0])

        # CLAHE
        img_clahe = image_processingv4.do_clahe(channels[1])

        # MORPHOLOGICAL FILTERS
        img_mf = image_processingv4.morph_filter(img_clahe, 3)

        # MASKING
        img_mask = image_processingv4.put_mask(image_processingv4.normalize(img_mf), mask)

        # HIGH BOOST FILTER
        img_hf = image_processingv4.normalize(image_processingv4.hb_filter(img_mask))

        # FRANGI
        img_fran = image_processingv4.normalize(image_processingv4.frangi_filter(img_hf))

        # THRESHOLD
        img_cont = image_processingv4.put_mask(image_processingv4.normalize(
            image_processingv4.contrast(img_fran, 90)), mask)
        kernel = np.ones((3, 3), np.uint8)
        img_cont = cv2.morphologyEx(img_cont, cv2.MORPH_OPEN, kernel)

        # HF PHOTO THRESHOLD
        img_sharp = image_processingv4.put_mask(
            unsharp_mask(img_hf, 15, 3), mask)
        th_1 = threshold_otsu(img_sharp)
        img_th = img_sharp <= th_1
        img_th = image_processingv4.put_mask(img_th, mask)
        kernel = np.ones((3, 3), np.uint8)
        img_th_mor = cv2.dilate(np.float32(img_th), kernel)
        img_th_mor = image_processingv4.morph_filter(img_th_mor, 3)
        kernel = np.ones((3, 3), np.uint8)
        img_th_mor = cv2.erode(np.float32(img_th_mor), kernel)

        # AND
        img_and = np.logical_and(img_cont, img_th_mor)
        img_and = image_processingv4.morph_filter(np.float32(img_and), 5)
        kernel = np.ones((5, 5), np.uint8)
        img_and = cv2.dilate(np.float32(img_and), kernel)
        kernel = np.ones((3, 3), np.uint8)
        img_and = cv2.erode(np.float32(img_and), kernel)

        return img_and


class KnnClassifier(Classifier):
    NAME = "K Nearest Neighbors"
    MODEL_FILE = os.path.join(Classifier.MODEL_DIR, 'knn')
    DEFAULT_PATCH_SIZE = 5
    DEFAULT_PATCH_COUNT = 10000
    DEFAULT_TRAINING_BALANCE = 0.5

    def __init__(self, patch_size=DEFAULT_PATCH_SIZE, patch_count=DEFAULT_PATCH_COUNT,
                 training_balance=DEFAULT_TRAINING_BALANCE):
        self.__model = KNeighborsClassifier()
        self.__patch_size = patch_size
        self.__patch_count = patch_count
        self.__training_balance = training_balance

    def train(self, train_generator):
        x, y = [], []
        for i, (image, mask) in enumerate(train_generator):
            point_generator = PointGeneratorBalanced(mask, self.__patch_count,
                                                     balance=self.__training_balance)
            feature_extractor = FeatureExtractor(image, self.__patch_size, point_generator)
            pixel_extractor = PixelExtractor(mask, point_generator)

            x.extend(feature_extractor.extract())
            y.extend(pixel_extractor.extract())

            stdout.write("\rTraining features: %.1f%%" % (100.0 * (i + 1) / train_generator.size))
            stdout.flush()
        stdout.write("\nTrain...")
        stdout.flush()
        self.__model.fit(x, y)
        stdout.write("\rTraining done\n")

    def _inner_classify(self, input_image):
        point_generator = PointGeneratorAll(input_image.shape)
        feature_extractor = FeatureExtractor(input_image, self.__patch_size, point_generator)
        return self.__model.predict([*feature_extractor.extract()]).reshape(input_image.shape)

    def save(self):
        Classifier.create_model_directory()
        joblib.dump(self.__model, KnnClassifier.MODEL_FILE)

    def load(self):
        self.__model = joblib.load(KnnClassifier.MODEL_FILE)


class NeuralNetworkClassifier(Classifier):
    NAME = "Neural Network"
    MODEL_FILE = os.path.join(Classifier.MODEL_DIR, 'nn')
    DEFAULT_PATCH_SIZE = 5
    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_EPOCHS = 5
    DEFAULT_TRAINING_BALANCE = 0.5

    def __init__(self, patch_size=DEFAULT_PATCH_SIZE, batch_size=DEFAULT_BATCH_SIZE,
                 epochs=DEFAULT_EPOCHS, training_balance=DEFAULT_TRAINING_BALANCE):
        self.__patch_size = patch_size
        self.__model = self.__build_model()
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__training_balance = training_balance

    def train(self, train_generator):
        def flip_generator(generator):
            for image, mask in generator:
                yield image, mask
                yield image[::-1, :], mask[::-1, :]
                yield image[:, ::-1], mask[:, ::-1]
                yield image[::-1, ::-1], mask[::-1, ::-1]

        def batch_generator(generator, epochs=1):
            for i in range(epochs):
                for image, mask in flip_generator(generator):
                    point_generator = PointGeneratorBalanced(mask, self.__batch_size,
                                                             balance=self.__training_balance)
                    patch_extractor = PatchExtractor(image, self.__patch_size, point_generator)
                    pixel_extractor = PixelExtractor(mask, point_generator)
                    patch_generator = zip(patch_extractor.extract(), pixel_extractor.extract())
                    x = []
                    y = []
                    for image_patch, mask_patch in patch_generator:
                        x.append(image_patch)
                        y.append(mask_patch)
                    yield np.array(x), np.array(y)

        train = batch_generator(train_generator, self.__epochs)
        self.__model.fit(train, batch_size=self.__batch_size,
                         steps_per_epoch=(4 * train_generator.size * self.__epochs))

    def _inner_classify(self, input_image):
        point_generator = PointGeneratorAll(input_image.shape)
        patch_extractor = PatchExtractor(input_image, self.__patch_size, point_generator)
        input_patches = np.array([*patch_extractor.extract()])
        stdout.write("\rPredict...")
        result = self.__model.predict(input_patches).reshape(input_image.shape[:2])
        result[result <= 0.5] = 0
        result[result > 0.5] = 1
        return result

    def save(self):
        Classifier.create_model_directory()
        self.__model.save_weights(NeuralNetworkClassifier.MODEL_FILE)

    def load(self):
        self.__model.load_weights(NeuralNetworkClassifier.MODEL_FILE)

    def __build_model(self):
        input_shape = (self.__patch_size, self.__patch_size, 1)
        filters = 8

        model = Sequential()

        model.add(Conv2D(filters, 3, activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(filters, 3, activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D())

        model.add(Conv2D(2 * filters, 3, activation='relu', padding='same'))
        model.add(Conv2D(2 * filters, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D())

        model.add(Conv2D(4 * filters, 3, activation='relu', padding='same'))
        model.add(Conv2D(4 * filters, 3, activation='relu', padding='same'))
        model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(4 * filters * (self.__patch_size // 8) ** 2, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                      metrics=[Precision(name='precision'), Recall(name='recall'), 'accuracy'])
        return model


class ClassifierFactory:
    @staticmethod
    def create_classifier(cls, load=False, train_generator=None, **kwargs):
        classifier = cls(**kwargs)
        if load:
            classifier.load()
        else:
            classifier.train(train_generator)
            classifier.save()
        return classifier
