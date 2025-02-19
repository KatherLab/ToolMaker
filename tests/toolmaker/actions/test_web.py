from collections.abc import Sequence

from pytest_mock import MockerFixture
from toolmaker.actions.web import (
    Browse,
    BrowseObservation,
    FileDownloadObservation,
    GoogleDriveDownloadFile,
    GoogleDriveListFolder,
    ListGoogleDriveFolderObservation,
    parse_html,
)


def test_browse(mocker: MockerFixture):
    # Mock the requests.get response
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.content = b"<html><body><p>Hello</p><p>World</p></body></html>"
    mock_get = mocker.patch("requests.get", return_value=mock_response)

    browse = Browse(url="https://example.com")
    result = browse()

    assert isinstance(result, BrowseObservation)
    assert result.status_code == 200
    assert result.content == "Hello World"
    mock_get.assert_called_once_with("https://example.com")


def test_parse_html_keeps_links(mocker: MockerFixture):
    html = "<html><body>This is a <a href='https://mydomain.com/abc'>link</a></body></html>"
    assert parse_html(html) == "This is a [link](https://mydomain.com/abc)"


def test_google_drive_list_folder():
    observation = GoogleDriveListFolder(
        url="https://drive.google.com/drive/folders/15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl",
        reasoning="",
    )()
    assert isinstance(observation, ListGoogleDriveFolderObservation)
    assert isinstance(observation.content, Sequence)
    assert len(observation.content) == 217
    assert (
        len(
            [
                f
                for f in observation.content
                if f.name == "deep_folder/folder_1/folder_1_1/file.txt"
            ]
        )
        == 1
    )


def test_google_drive_download_file(tmp_path_factory):
    output_path = tmp_path_factory.mktemp("google_drive") / "egg.txt"
    observation = GoogleDriveDownloadFile(
        url="https://drive.google.com/file/d/1NEVRByyKxcUCdyqLU4UCYoHaMaOx4GUE/view?usp=share_link",
        output_path=str(output_path),
    )()
    assert isinstance(observation, FileDownloadObservation)
    assert observation.path == str(output_path.resolve().absolute())
    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.read_text().strip() == "egg"
