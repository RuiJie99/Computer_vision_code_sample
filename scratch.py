from typing import List, Dict, Set, Iterable, Callable, Tuple
from pathlib import Path
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import cv2 as cv
from sklearn import neighbors, datasets
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import cluster
from sklearn.linear_model import LogisticRegression




#size of data
class DsetSize(Enum):
    Big = 'big'
    Small = 'small'
#type of data
class DsetType(Enum):
    Train = 'train'
    Test = 'test'

#function to load image from path
# path_to_image contains : iterator with path; returns a list of np array

cats = "cat"
dogs = "dog"

def get_all_image_paths(size: DsetSize, type: DsetType, animal: str) -> Iterable[Path]:
    path_of_images = None
    if(size == DsetSize.Small):
        #TODO: get small dataset
        if(type == DsetType.Train):
            #TODO: get training data
            if(animal == "cat"):
                #TODO: get cats
                path_of_images = Path("02-cats-n-dogs-master/data/small/train/cats")
            else:
                #TODO: get dogs
                path_of_images = Path("02-cats-n-dogs-master/data/small/train/dogs")
        else:
            #TODO: get test Data
            if(animal == "cat"):
                #TODO: get cats
                path_of_images = Path(
                    "02-cats-n-dogs-master/data/small/test/cats")
            else:
                #TODO: get dogs
                path_of_images = Path(
                    "02-cats-n-dogs-master/data/small/test/dogs")
    else:
        #TODO: get big dataset
        if(type == DsetType.Train):
            #TODO: get training data
            if(animal == "cat"):
                #TODO: get cats
                path_of_images = Path("02-cats-n-dogs-master/data/big/train/cats")
            else:
                #TODO: get dogs
                path_of_images = Path("02-cats-n-dogs-master/data/big/train/dogs")
        else:
            #TODO: get test Data
            if(animal == "cat"):
                #TODO: get cats
                path_of_images = Path(
                    "02-cats-n-dogs-master/data/big/test/cats")
            else:
                #TODO: get dogs
                path_of_images = Path(
                    "02-cats-n-dogs-master/data/big/test/dogs")
    #TODO: convert list of strings to Iterable[Path]
    #
    list_files = (Path(filename) for filename in (path_of_images).glob('*.jpg'))
    return list_files

def load_images(path_to_images: Iterable[Path]) -> List[np.ndarray]:
    """Hint: use skimage"""
    # TODO : load all the images
    list_of_images = [io.imread(path) for path in path_to_images]
    return list_of_images

def load_dataset(size: DsetSize, type: DsetType) -> Tuple[List[np.ndarray], List[str]]:
    """
    Return a dataset
    :return: a tuple of (images, labels)
    """
    # TODO
    cats_paths = None
    dogs_paths = None
    if(size == DsetSize.Small):
        #TODO: get small dataset
        if(type == DsetType.Train):
            #TODO: get training data
            #TODO: get cats
            cats_paths = get_all_image_paths(DsetSize.Small, DsetType.Train, cats)
            dogs_paths = get_all_image_paths(DsetSize.Small, DsetType.Train, dogs)
        else:
            #TODO: get test Data
            cats_paths = get_all_image_paths(DsetSize.Small, DsetType.Test, cats)
            dogs_paths = get_all_image_paths(DsetSize.Small, DsetType.Test, dogs)
    else:
        #TODO: get big dataset
        print("big")
        if(type == DsetType.Train):
            print("train")
            #TODO: get training data
            cats_paths = get_all_image_paths(DsetSize.Big, DsetType.Train, cats)
            dogs_paths = get_all_image_paths(DsetSize.Big, DsetType.Train, dogs)
        else:
            #TODO: get test Data
            cats_paths = get_all_image_paths(DsetSize.Big, DsetType.Test, cats)
            dogs_paths = get_all_image_paths(DsetSize.Big, DsetType.Test, dogs)
    list_dogs = load_images(dogs_paths)
    list_cats = load_images(cats_paths)
    labels = ["dog"]*len(list_dogs) + ["cat"]*len(list_cats)
    return (list_dogs + list_cats, labels)

class ImgFeaturizerABC(ABC):
    def fit(self, images, labels):
        return self
    def transform(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [
            self._transform_one(img)
            for img in images
        ]
    @abstractmethod
    def _transform_one(self, image: np.ndarray) -> np.ndarray:
        pass

@dataclass
class RawFeaturizer(ImgFeaturizerABC):
    """
    This featurizer simply use the raw pixels
    """
    size: Tuple[int, int] = (64, 64)
    def _transform_one(self, img: np.ndarray) -> np.ndarray:
        """
        Resize then flatten the RBG pixel intensities into a single vectors of numbers.
        Hint: Use skimage to resize, and numpy to flatten
        """
        # TODO
        imageArray = transform.resize(img, self.size)
        return np.ndarray.flatten(imageArray)

@dataclass
class HistogramFeaturizer(ImgFeaturizerABC):

    bins: Tuple[int, int, int] = (8, 8, 8)
    def _transform_one(self, image: np.ndarray) -> np.ndarray:
        #TODO: convert from RGB (Red, Green, Blue) to HSV (Hue, Saturation, and Value)
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        ##TODO: calculate histogram for [image], with channel [0,1,2], with [8,8,8]
        ## intervals for each channel, values ranging from 0-255 for each one
        hist = cv.calcHist([image], [0,1,2], None, self.bins, [0,256,0,256,0,256], accumulate=False)
        return np.ndarray.flatten(hist)

def nearestNeignbour():
    ## TODO: define and train a pipeline using the raw featurizer and a k-NN classifer
    ## TODO: score it on the test set
    ##      1. Get all dogs and cats images (images: List[np.ndarray] et nom: List[str])
    ##      2. Use histogram featurizer to transform images
    ##      3. define hyperparameters
    ##      4. GridSearchCV
    print('start')
    big_train_images, big_train_labels = load_dataset(DsetSize.Big, DsetType.Train)
    x =  big_train_images
    y = big_train_labels
    print('finished loading')
    knn = neighbors.KNeighborsClassifier()
    hist_feature = HistogramFeaturizer()
    pipe = Pipeline(steps=[('transform', hist_feature),
                     ('k-NearestNeighbours', knn)])

    hyper_parameters = {
        'k-NearestNeighbours__n_neighbors':[i for i in range(1,20)]
    }
    print('hyperparameters compile')
    search = GridSearchCV(
        pipe,
        hyper_parameters
    )
    print('gridsearchCV compiles')
    search.fit(x,y)
    print('done fitting')
    test_array_img, names = load_dataset(DsetSize.Big, DsetType.Test)
    plt.imshow(test_array_img[700])
    plt.show()
    nbTrue = 0
    nbFalse = 0
    print(test_array_img)
    predicted = search.predict(test_array_img)
    for i in range(0, int(len(test_array_img) / 2)):
        predictedElement = predicted[i]
        actualElement = names[i]
        if predictedElement == actualElement:
            nbTrue += 1
        else:
            nbFalse += 1
    percentageTrue = nbTrue/(nbFalse + nbTrue)

    print(f"Raw pixel representation accuracy: ", percentageTrue)

def test():
    nearestNeignbour()

test()