import re
import requests
import hashlib
import os, os.path as osp


def check_url_exists(url):
    try:
        requests.get(url)
        return True
    except:
        return False


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_google_file_index(file_url):
    if '/' not in file_url:
        return file_url
    else:
        url_parts = file_url.split("/")
        assert 'https' in file_url and 'drive.google.com' in file_url and 'file' in file_url and 'd' in file_url
        # Assume url has format https://drive.google.com/file/d/XXXXXX?????XXXXXXX/view?usp=sharing
        return url_parts[url_parts.index('d') + 1]


def md5sum(filename, block_size=None):
    if block_size is None:
        block_size = 65536
    hash_res = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hash_res.update(block)
    return hash_res.hexdigest()


def check_md5sum(filename, md5, block_size=None):
    if not (isinstance(md5, str) and len(md5) == 32):
        raise ValueError(f"MD5 must be 32 chars: {md5}")

    # print(f"Computing MD5: {filename}")
    md5_actual = md5sum(filename, block_size=block_size)

    if md5_actual == md5:
        # print(f"MD5 matches: {filename}")
        return True
    else:
        print(f"MD5 does not match!: {filename} has md5 {md5_actual}, target md5 is {md5}")
        return False


def download_file_from_google_drive(file_url, destination, cached=False, md5=None):
    if file_url is None:
        return
    file_index = get_google_file_index(file_url)
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_index}, stream=True)
    print(response.cookies.items())
    token = get_confirm_token(response)
    if token:
        params = {'id': file_index, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    print(response)

    if "Content-Disposition" in response.headers and destination[-1] == '/':
        m = re.search('filename="(.*)"', response.headers["Content-Disposition"])
        filename_from_url = m.groups()[0]
        destination = osp.join(destination, filename_from_url)
    assert destination[-1] != '/'
    if not osp.exists(osp.dirname(destination)):
        print(f'Create folder {osp.dirname(destination)}')
        os.makedirs(osp.dirname(destination))

    if cached and osp.exists(destination):
        sign = True
        if md5 is not None:
            sign = check_md5sum(destination, md5)
        if sign:
            print(f'File {destination} exists! Download nothing from google drive!')
            return
    save_response_content(response, destination)
    if md5 is not None and check_md5sum(destination, md5):
        print("MD5 does not match !!!!")
        return
    file_size = osp.getsize(destination) / (1024 ** 2)
    print(f'Finish download {destination}, File size {file_size:.2f}MB')
    return file_size

