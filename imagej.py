#! /usr/bin/env python2

import sys
import os
import re
import requests
if sys.version_info.major == 2:
    from urlparse import urljoin
else:
    from urllib.parse import urljoin

__version__ = '0.1.0'

HOST = 'http://localhost:8080'

"""wrapper for imagej-server APIs"""


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
    """Gets a list of information for objects available on imagej-server.

    API: GET /objects

    :param host: host address of imagej-server
    :return: a list of object information
    :rtype: list[string]
    """
    url = urljoin(host, 'objects')
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def get_object(id, host=HOST):
    """Shows the information of an object.

    API: GET /objects/{id}

    :param id: object ID to show information
    """

    url = urljoin(host, 'objects/%s' % id)
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def remove_object(id, host=HOST):
    """Removes one object form imagej-server.

    API: DELETE /objects/{id}

    :param id: object ID to remove
    """

    url = urljoin(host, 'objects/%s' % id)
    r = requests.delete(url)
    r.raise_for_status()


def upload_file(data, type=None, host=HOST):
    """Uploads a file to imagej-server (currently only supports image and table
    in text).

    API: POST /objects/upload?[type=TYPE]

    :param data: file-like object to be uploaded
    :param type: hint for file type
    :param host: host address of imagej-server
    :return: object ID representing the file in form of
            {"id": "object:1234567890abcdef"}
    :rtype: dict
    """

    url = urljoin(host, 'objects/upload')
    if type:
        url += '?type=' + type
    r = requests.post(url, files={'file': data})
    r.raise_for_status()
    return r.json()


def retrieve_object(id, format, config=None, host=HOST):
    """Retrieves an object as a file in specific format

    API: POST /objects/{id}/{format}?[&key=value]...

    :param id: object ID
    :param format: format of the object to be stored in
    :param config: configuration for storing the file (not tested)
    :param host: host address of imagej-server
    :return: filename for downloading the requested file in the form of
            {"filename": "asdf1234.png"}
    :rtype: dict
    """

    url = urljoin(host, 'objects/%s/%s' % (id, format))
    r = requests.get(url, params=config)
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


class IJ(object):

    """Basic client for imagej-server."""

    def __init__(self, host=HOST):
        """Creates a client that bounds to host.

        :param host: address of imagej-server
        """
        self._modules = None
        self._objects = None
        self._host = None
        self.host = host

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, value):
        """Sets the host of IJ to a new value.

        Adds schema and/or port to value if necessary. Also checks if
        connection can be established.
        """
        if value is not None:
            m = re.match('(.+?://)?.+?(:\d{1,5})?$', value)
            if m.group(1) is None:
                print('No schema supplied for host address. Using "http".')
                value = 'http://' + value
            if m.group(2) is None:
                print('No port supplied for host address. Using "8080".')
                value = value + ':8080'
        else:
            value = HOST

        oldhost, self._host = self._host, value
        try:
            self.modules(refresh=True)
        except:
            self._host = oldhost
            raise RuntimeError('Cannot connect to host: %s' % value)

    def modules(self, refresh=False):
        """Gets the module IDs of imagej-server if no cache is available or
        refresh is set to True, or returns the cache for the IDs otherwise.

        :param refresh: force fetching modules from imagej-server if True
        :return: imagej-server module IDs
        :rtype: list[string]
        """

        if self._modules is None or refresh:
            self._modules = get_modules(self.host)
        return self._modules

    def find(self, regex):
        """Finds all module IDs that match the regular expression.

        :param regex: the regular express to match the module IDs
        :return: all matching IDs
        :rtype: list[string]
        """

        pattern = re.compile(regex)
        return list(filter(pattern.search, self.modules()))

    def detail(self, id):
        """Gets the detail of a module or an object specified by the ID.

        :param id: the ID of a module or an object
        :return: details of a module or an object
        :rtype: dict
        """

        if id.startswith('object:'):
            return get_object(id, self.host)
        return get_module(id, self.host)

    def run(self, id, inputs=None, process=True):
        """Runs a module specified by the ID with inputs.

        :param id: the ID of the module
        :param inputs: a dict-like object containing inputs for the execution
        :param process: if the execution should be pre/post processed
        :return: outputs of the execution
        :rtype: dict
        """

        return run_module(id, inputs, process, self.host)

    def objects(self):
        """Gets a list of sorted object IDs being served on imagej-server.

        :return: a list of object IDs
        :rtype: list[string]
        """
        return sorted(get_objects(host=self.host))

    def remove(self, id):
        """Removes one object from imagej-server.

        :param id: object ID to remove
        """

        remove_object(id, self.host)

    def upload(self, filename, type=None):
        """Uploads a file to imagej-server

        :param filename: filename of the file to be uploaded
        :param type: optional hint for file type
        :return: object ID of the uploaded file
        :rtype: string
        """

        with open(filename, 'rb') as data:
            return upload_file(data, type, self.host)['id']

    def retrieve(self, id, format='tif', config=None, dest=None):
        """Retrieves an object in specific format from imagej-server.

        If dest is None, the raw content would be returned.

        :param id: object ID
        :param format: file format the object to be saved into
        :param config: configuration for storing the object (not tested)
        :param dest: download destination
        :return: content of the object if dest is None, otherwise None
        :rtype: string or None
        """

        content = retrieve_object(id, format, config, host=self.host)
        if dest is None:
            return content
        if os.path.isdir(dest):
            dest = os.path.join(dest, id[len('object:'):] + '.' + format)
        else:
            dir = os.path.dirname(dest)
            if not os.path.isdir(dir):
                raise Exception('Directory does not exist: %s' % dir)
        if os.path.isfile(dest):
            print('Overwriting existed file: %s' % dest)
        with open(dest, 'wb') as f:
            f.write(content)

    def show(self, id, format='tif', config=None):
        """Retrieves and shows an object in specific format from imagej-server.

        PIL is needed for this function. In addition, image viewing software
        must exist on the system (i.e. display or xv on Unix).

        :param id: object ID if format is set, or a file being served
        :param format: file format the object to be saved into
        :param config: configuration for storing the object (not tested)
        """

        from PIL import Image
        import io

        content = retrieve_object(id, format, config, host=self.host)
        Image.open(io.BytesIO(content)).show()
