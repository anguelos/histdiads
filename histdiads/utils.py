import tqdm
import os
import requests
import pathlib
import tarfile
import zipfile

def download_url(url, filename, filesize=None, resume=True):
    pathlib.Path(filename).parents[0].mkdir(parents=True, exist_ok=True)
    if filesize is not None and os.path.isfile(filename) and os.path.getsize(filename) == filesize:
        print(f"Found {filename} cached")
        return

    if filesize is not None and os.path.isfile(filename) and resume:
        found_bytes_count = os.path.getsize(filename)
        resume_header = {"Range": f"bytes={found_bytes_count}"}
        response = requests.get(url, headers=resume_header, stream=True,  verify=False, allow_redirects=True)
        file_mode="ab"
    else:
        response = requests.get(url, stream=True, verify=False, allow_redirects=True)
        file_mode="wb"
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024**2  # 1 MB
    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(filename, file_mode) as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    #if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
    #    print(f"ERROR, something went wrong downloading {url} to {filename}")


def extract(compressed_path, root):
    if compressed_path.name.endswith(".tar.gz"):
        my_tar = tarfile.open(compressed_path,"r:gz")
        my_tar.extractall(root)
        my_tar.close()
    elif compressed_path.name.endswith(".zip"):
        archive = zipfile.ZipFile(compressed_path)
        archive.extractall(root)
        archive.close()
    else:
        raise ValueError