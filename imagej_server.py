"""Wrapper for imagej-server APIs"""

import requests
from urlparse import urljoin


HOST = 'http://localhost:8080'


def get_modules(host=HOST):
    """Gets a list of module IDs.

    API: GET /modules

    :param host: host address of imagej-server
    :return: a list of module IDs
    :rtype: list of string
    """

    url = urljoin(host, 'modules')
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def get_module(id, host=HOST):
    """Gets the detail of one module.

    API: GET /modules/{id}

    :param id: module ID
    :param host: host address of imagej-server
    :return: details of module specified by ID
    :rtype: dict
    """

    url = urljoin(host, 'modules/%s' % id)
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def run_module(id, inputs=None, process=True, host=HOST):
    """Runs a module with inputs.

    API: POST /modules/{id}?[process=(true,false)]

    :param id: module ID
    :param inputs: inputs for module execution
    :param process: if the execution should be pre/post processed
    :param host: host address of imagej-server
    :return: outputs of the execution
    :rtype: dict
    """

    url = urljoin(host, 'modules/%s?process=%s' %
                  (id, str(process).lower()))
    r = requests.post(url, json=inputs)
    r.raise_for_status()
    return r.json()


def get_objects(host=HOST):
    """Gets a list of object IDs available on imagej-server.

    API: GET /io/objects

    :param host: host address of imagej-server
    :return: a list of object IDs
    :rtype: list[string]
    """
    url = urljoin(host, 'io/objects')
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def get_files(host=HOST):
    """Gets a list of files being served on imagej-server.

    API: GET /io/files

    :param host: host address of imagej-server
    :return: a list of filenames
    :rtype: list[string]
    """
    url = urljoin(host, 'io/files')
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def upload_file(data, host=HOST):
    """Uploads a file to imagej-server (currently only supports image files).

    API: POST /io/file

    :param data: file-like object to be uploaded
    :param host: host address of imagej-server
    :return: object ID representing the file in form of
            {"id": "object:1234567890abcdef"}
    :rtype: dict
    """

    url = urljoin(host, 'io/file')
    r = requests.post(url, files={'file': data})
    r.raise_for_status()
    return r.json()


def request_file(id, format, config=None, host=HOST):
    """Requests to stores an object as a file in specific format for
    download.

    API: POST /io/file/{id}?format=FORMAT

    :param id: object ID
    :param format: format of the object to be stored in
    :param config: configuration for storing the file (not tested)
    :param host: host address of imagej-server
    :return: filename for downloading the requested file in the form of
            {"filename": "asdf1234.png"}
    :rtype: dict
    """

    url = urljoin(host, 'io/file/%s' % id)
    r = requests.post(
        url, params={'format': format}, json={'config': config})
    r.raise_for_status()
    return r.json()


def retrieve_file(filename, host=HOST):
    """Retrieves the content of a file.

    API: GET /io/file/{filename}

    :param filename: name of file to be download
    :param host: host address of imagej-server
    :return: the downloaded file
    :rtype: file
    """

    url = urljoin(host, 'io/file/%s' % filename)
    r = requests.get(url)
    r.raise_for_status()
    return r.content


def stop(host=HOST):
    """Stops the imagej-server gracefully.

    API: DELETE /admin/stop

    :param host: host address of imagej-server
    """

    url = urljoin(host, 'admin/stop')
    r = requests.delete(url)
    r.raise_for_status()
