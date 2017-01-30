#! /usr/bin/env python2

from imagej_server import *
import os
import re


class Client(object):

    """Basic client for imagej-server."""

    def __init__(self, host=HOST):
        self.host = HOST
        self._modules = None

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
        return filter(pattern.search, self.modules())

    def detail(self, id):
        """Gets the detail of a module specified by the ID.

        :param id: the ID of the module
        :return: details of a module
        :rtype: dict
        """

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
        """Gets a list of objects being served on imagej-server, sorted by ID.

        :return: a list of object IDs
        :rtype: list[string]
        """
        return sorted(get_objects(host=self.host))

    def upload(self, filename):
        """Uploads a file to imagej-server

        :param filename: filename of the file to be uploaded
        :return: object ID of the uploaded file
        :rtype: string
        """

        with open(filename, 'rb') as data:
            return upload_file(data, self.host)['id']

    def retrieve(self, id, format, config=None, dest=None):
        """Retrieves an object in specific format from imagej-server.

        If dest is None, the raw content would be returned.

        :param id: object ID
        :param format: file format the object to be saved into
        :param config: configuration for storing the object (not tested)
        :param dest: download destination
        :return: content of the object if dest is None, otherwise None
        :rtype: str or None
        """

        content = retrieve_file(id, format, config, host=self.host)
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

    def show(self, id, format, config=None):
        """Retrieves and shows an object in specific format from imagej-server.

        :param id: object ID if format is set, or a file being served
        :param format: file format the object to be saved into
        :param config: configuration for storing the object (not tested)
        """

        from PIL import Image
        import io

        content = retrieve_file(id, format, config, host=self.host)
        Image.open(io.BytesIO(content)).show()