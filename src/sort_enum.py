from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import cv2
from PIL import Image
import rasterfairy


def sort_by_date_asc(album_list, album_art) -> tuple[list, list]:
    combined = list(zip(album_list, album_art))

    # Sort by album_list's 'date'
    combined_sorted = sorted(combined, key=lambda pair: pair[0]["date"])

    album_list, album_art = zip(*combined_sorted)

    return album_list, album_art


def sort_by_color(album_list, album_art) -> tuple[list, list]:
    all_features = []

    for album in album_art:
        album = album.resize((64, 64), Image.Resampling.BICUBIC)
        if album.mode == "RGB":
            img = np.array(album)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        else:
            img = np.array(album)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        all_features += [np.array(img).ravel()]
    all_features = np.vstack(all_features)

    all_features = MinMaxScaler().fit_transform(all_features)
    all_features = StandardScaler().fit_transform(all_features)

    embedding = TSNE().fit_transform(all_features)

    assignment = rasterfairy.transformPointCloud2D(
        embedding,
    )
    idx = np.lexsort((assignment[0][:, 0], assignment[0][:, 1]))

    album_list = [album_list[i] for i in idx]
    album_art = [album_art[i] for i in idx]

    return album_list, album_art


class SortMode(Enum):
    BY_DATE = sort_by_date_asc
    BY_COLOR = sort_by_color

    def __call__(self, album_list, album_art):
        return self.value(album_list, album_art)
