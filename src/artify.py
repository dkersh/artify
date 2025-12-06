from spotipy import Spotify
from PIL import Image
import numpy as np
from typing import Optional
import requests
from io import BytesIO
from tqdm import tqdm
from operator import itemgetter
from sort_enum import SortMode


class Artify:
    def __init__(
        self,
        sp_client: Spotify,
        albums_list: list[str] = [],
        album_art: list[Image] = [],
    ):
        self.sp_client = sp_client
        self.albums_list = albums_list
        self.album_art = []
        self.offset = 0

    def get_top_albums(self, batch_size: int = 25, N: int = 50) -> list:
        unique_albums = []
        with tqdm(
            total=N, desc=f"Pulling top albums with batch_size {batch_size}"
        ) as pbar:
            while len(unique_albums) < N:
                top_tracks = self.sp_client.current_user_top_tracks(
                    limit=batch_size, offset=offset
                )

                for track in top_tracks["items"]:
                    album = track["album"]
                    album_dict = {
                        "id": album["id"],
                        "artist": album["artists"][0]["name"],
                        "name": album["name"],
                        "date": album["release_date"],
                        "art": album["images"][0][
                            "url"
                        ],  # We will pull album art later
                    }
                    if is_unique(album_dict, unique_albums):
                        unique_albums += [album_dict]
                        pbar.update(1)
                    if len(unique_albums) == N:
                        break
                self.offset += batch_size

        self.albums_list = unique_albums

    def download_albums(self):
        if len(self.albums_list) == 0:
            raise Exception("Pull top albums first please")

        for album in tqdm(self.albums_list, desc="Downloading album art"):
            self.album_art += [download_album_art(album["art"])]

    def sort_albums(self, ordering: SortMode = SortMode.BY_DATE):
        self.albums_list, self.album_art = ordering(self.albums_list, self.album_art)

    def generate_mosaic(self, resolution: tuple | list = (1000, 1000)):
        sides = int(np.sqrt(len(self.albums_list)))

        mosaic = Image.new(
            "RGB", (sides * resolution[0], sides * resolution[1]), (0, 0, 0)
        )

        x, y = 0, 0
        for im in tqdm(self.album_art, desc="Generating Album Mosaic"):
            mosaic.paste(
                im.resize((resolution[0], resolution[1]), Image.Resampling.BICUBIC),
                (x, y),
            )
            x += resolution[0]
            if x >= sides * resolution[0]:
                x = 0
                y += resolution[1]

        return mosaic


def download_album_art(img_url: str):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    return img


def is_unique(album_dict: dict, unique_albums: list[dict]) -> bool:
    for album in unique_albums:
        if album_dict["id"] == album["id"]:
            return False
    return True
