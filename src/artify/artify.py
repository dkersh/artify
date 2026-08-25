from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import spotipy
from joblib import Parallel, delayed
from PIL import Image
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

from artify import image_feature_processing


@dataclass(frozen=True)
class AlbumArt:
    id: str
    artist_name: str = field(compare=False)
    album_name: str = field(compare=False)
    image_url: str = field(compare=False)


class Artify:
    def __init__(
        self,
        client_id,
        client_secret,
        redirect_uri,
        cache_dir: Path | str = "./.cache_art",
    ):
        """Artify Class for Pulling top albums and generating an album mosaic.

        Args:
            client_id (_type_): Spotify API client ID key.
            client_secret (_type_): Spotify API secret key.
            redirect_uri (_type_): Spotify API redirect URL.
            cache_dir (Path | str, optional): Directory to cache album art to.
                Defaults to "./.cache_art".
        """
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-top-read",
            )
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.top_albums: list[AlbumArt] | None = None

    def pull_top_albums(self, N: int, batch_size: int = 50) -> list[AlbumArt]:
        """Algorithm for pulling listener's top N albums.

        TODO: Skip albums which have the same album art.

        Args:
            N (int): Number of albums to pull - should be a square number.
            batch_size (int, optional): Batch size to query with. Defaults to 50.

        Raises:
            ValueError: Error if no top tracks identified.

        Returns:
            list[AlbumArt]: List of listener's top spotify albums.
        """
        all_albums = []
        offset = 0
        while len(all_albums) <= N:
            top_tracks = self.sp.current_user_top_tracks(
                time_range="long_term", limit=batch_size, offset=offset
            )

            if top_tracks is None:
                raise ValueError("No Top Tracks obtained")

            for track in top_tracks["items"]:
                album = track["album"]

                album = AlbumArt(
                    id=album["id"],
                    artist_name=album["artists"][0]["name"],
                    album_name=album["name"],
                    image_url=album["images"][0]["url"],
                )

                if album not in all_albums:
                    all_albums += [album]

                    if len(all_albums) == N:
                        self.top_albums = all_albums

                        return self.top_albums
            offset += batch_size

        self.top_albums = all_albums

        return self.top_albums

    def _download_single_album_art(self, album: AlbumArt) -> Image.Image:
        """Method to download album art.

        Args:
            album (AlbumArt): Input AlbumArt object.

        Returns:
            Image.Image: Pillow image of album.
        """
        album_file_path = self.cache_dir / f"{album.id}.png"

        response = requests.get(album.image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        img.save(album_file_path)

        return img

    def download_album_art(self) -> list[Image.Image]:
        """Download all album art in parallel.

        Raises:
            ValueError: Fails if no top albums have been pulled yet.

        Returns:
            list[Image.Image]: List of album art pillow images.
        """
        if self.top_albums is None:
            raise ValueError("No top albums founds")

        all_album_imgs = Parallel(n_jobs=-1, verbose=10, backend="threading")(
            delayed(self._download_single_album_art)(album) for album in self.top_albums
        )

        return all_album_imgs

    def generate_mosaic(self, resolution: tuple[int, int] = (100, 100)) -> Image.Image:
        """Generate a mosaic of album art, sorted by colour.

        Args:
            resolution (tuple[int, int], optional): Resolution of each album art.
                Defaults to (100, 100).

        Raises:
            ValueError: Will fail if no top albums have been pooled.

        Returns:
            Image.Image: Mosaic image.
        """
        if self.top_albums is None:
            raise ValueError("No Top Albums obtained.")

        sides = int(np.sqrt(len(self.top_albums)))

        mosaic = Image.new(
            "RGB", (sides * resolution[0], sides * resolution[1]), (0, 0, 0)
        )

        all_album_imgs = self.download_album_art()

        # Sort albums by colour
        sorted_album_imgs = image_feature_processing.sort_images_by_colour(
            all_album_imgs
        )

        x, y = 0, 0
        for im in tqdm(sorted_album_imgs, desc="Generating Album Mosaic"):
            if im is None:
                raise ValueError("Not all album art has been found.")
            mosaic.paste(
                im.resize((resolution[0], resolution[1]), Image.Resampling.BICUBIC),
                (x, y),
            )
            x += resolution[0]
            if x >= sides * resolution[0]:
                x = 0
                y += resolution[1]

        return mosaic
