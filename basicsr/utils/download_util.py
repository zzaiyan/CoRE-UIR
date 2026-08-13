# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import math
import os
import requests
from tqdm import tqdm

from .misc import sizeof_fmt


def resolve_checkpoint_path(path):
    """Resolve local paths and ``hf://`` checkpoint URIs to local files.

    Hugging Face URIs use ``hf://owner/repo/filename`` and may pin a revision
    with ``hf://owner/repo@revision/filename``.
    """
    if path is None:
        return None
    path = os.fspath(path)
    if not path.startswith('hf://'):
        return os.path.expanduser(path)

    components = path[len('hf://'):].split('/', 2)
    if len(components) != 3 or not all(components):
        raise ValueError(
            'Hugging Face checkpoint URI must use '
            'hf://<owner>/<repo>[@<revision>]/<filename>.')

    owner, repo_revision, filename = components
    if '@' in repo_revision:
        repo, revision = repo_revision.rsplit('@', 1)
        if not repo or not revision:
            raise ValueError(f'Invalid Hugging Face checkpoint URI: {path}')
    else:
        repo, revision = repo_revision, None

    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=f'{owner}/{repo}',
        filename=filename,
        revision=revision)


def download_file_from_google_drive(file_id, save_path):
    """Download files from google drive.

    Ref:
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive  # noqa E501

    Args:
        file_id (str): File id.
        save_path (str): Save path.
    """

    session = requests.Session()
    URL = 'https://docs.google.com/uc?export=download'
    params = {'id': file_id}

    response = session.get(URL, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params['confirm'] = token
        response = session.get(URL, params=params, stream=True)

    # get file size
    response_file_size = session.get(
        URL, params=params, stream=True, headers={'Range': 'bytes=0-2'})
    if 'Content-Range' in response_file_size.headers:
        file_size = int(
            response_file_size.headers['Content-Range'].split('/')[1])
    else:
        file_size = None

    save_response_content(response, save_path, file_size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response,
                          destination,
                          file_size=None,
                          chunk_size=32768):
    if file_size is not None:
        pbar = tqdm(total=math.ceil(file_size / chunk_size), unit='chunk')

        readable_file_size = sizeof_fmt(file_size)
    else:
        pbar = None

    with open(destination, 'wb') as f:
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size):
            downloaded_size += chunk_size
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f'Download {sizeof_fmt(downloaded_size)} '
                                     f'/ {readable_file_size}')
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        if pbar is not None:
            pbar.close()
