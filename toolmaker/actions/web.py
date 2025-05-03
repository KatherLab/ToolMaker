import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Iterator

import gdown
import requests
from bs4 import BeautifulSoup
from gdown.download import _get_session as gdown_get_session
from gdown.download_folder import MAX_NUMBER_FILES as GDOWN_MAX_NUMBER_FILES
from gdown.download_folder import (
    _download_and_parse_google_drive_link as gdown_download_and_parse_google_drive_link,
)
from gdown.download_folder import _GoogleDriveFile as GdownGoogleDriveFile
from gdown.exceptions import FileURLRetrievalError, FolderContentsMaximumLimitError
from loguru import logger
from pydantic import BaseModel, Field

from toolmaker.actions.actions import Action, Observation, register_action
from toolmaker.actions.errors import FunctionCallError


class BrowseObservation(Observation):
    status_code: int
    content: str


class FileDownloadObservation(Observation):
    path: str


class GoogleDriveFile(BaseModel):
    name: str
    url: str


class ListGoogleDriveFolderObservation(Observation):
    content: Sequence[GoogleDriveFile]


def parse_html(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")

    # Convert links to markdown format while keeping them in the text
    for link in soup.find_all("a"):
        if link.get("href"):
            link.replace_with(f"[{link.get_text().strip()}]({link.get('href')})")

    # Get all text content with preserved markdown-formatted links
    return " ".join(soup.stripped_strings)


@register_action
class Browse(Action):
    """Fetch a URL and return the content."""

    action = "browse"
    url: str = Field(..., description="The URL to open.")

    def __call__(self) -> BrowseObservation:
        logger.info(f"Browsing {self.url}")
        try:
            response = requests.get(self.url)
            try:
                content = parse_html(response.content)
            except Exception:
                logger.warning(
                    f"Failed to parse HTML for {self.url}, using raw content"
                )
                content = response.content
            return BrowseObservation(status_code=response.status_code, content=content)
        except requests.exceptions.RequestException as e:
            raise FunctionCallError(str(e))

    bash_side_effect = False

    def bash(self) -> str:
        return f"wget {shlex.quote(self.url)}"


@register_action
class GoogleDriveListFolder(Action):
    """List the files in a Google Drive folder."""

    action = "google_drive_list_folder"
    url: str = Field(..., description="The URL of the Google Drive folder.")

    def __call__(self) -> ListGoogleDriveFolderObservation:
        # Use gdown's internal functions to get folder structure
        sess = gdown_get_session(use_cookies=True, proxy=None, user_agent=None)
        success, gdrive_file = gdown_download_and_parse_google_drive_link(
            sess=sess,
            url=self.url,
            quiet=False,
            remaining_ok=True,
            verify=True,
        )

        if not success:
            raise FunctionCallError("Failed to retrieve folder contents")

        def format_tree(
            file: GdownGoogleDriveFile, parent: str = ""
        ) -> Iterator[GoogleDriveFile]:
            for child in file.children:
                name = f"{parent}{child.name}" if parent else child.name
                name += "/" if child.is_folder() else ""
                if child.is_folder():
                    # yield GoogleDriveFile(
                    #     name=name,
                    #     url=f"https://drive.google.com/drive/folders/{child.id}",
                    # )
                    yield from format_tree(child, parent=name)
                else:
                    yield GoogleDriveFile(
                        name=name,
                        url=f"https://drive.google.com/uc?id={child.id}",
                    )

        try:
            return ListGoogleDriveFolderObservation(
                content=list(format_tree(gdrive_file))
            )
        except FolderContentsMaximumLimitError:
            raise FunctionCallError(
                f"The folder contains too many files to list (maximum {GDOWN_MAX_NUMBER_FILES})"
            )

    bash_side_effect = False

    def bash(self) -> str:
        return f"uvx gdown --list {shlex.quote(self.url)}"  # TODO: this is not a valid command


@register_action
class GoogleDriveDownloadFile(Action):
    """Download a file from Google Drive. Note that this actions does not work with folders."""

    action = "google_drive_download_file"
    url: str = Field(..., description="The URL of the Google Drive file to download.")
    output_path: str = Field(..., description="The path to save the downloaded file.")

    def __call__(self) -> FileDownloadObservation:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(self.url, self.output_path, quiet=False, fuzzy=True)
        except FileURLRetrievalError as e:
            raise FunctionCallError(
                f"Failed to retrieve file URL: {self.url}. NOTE: It may be the case that you are trying to download a folder instead of a file. Perhaps you should use the `google_drive_list_folder` action to list the files in the folder (which will help you identify if this is the case) and then use the `google_drive_download_file` action to download the files individually. Full error message: {e!s}"
            )

        return FileDownloadObservation(
            content=None,
            path=str(Path(self.output_path).resolve().absolute()),
        )

    bash_side_effect = True

    def bash(self) -> str:
        return f"uvx gdown --fuzzy {shlex.quote(self.url)} -O {shlex.quote(self.output_path)}"
