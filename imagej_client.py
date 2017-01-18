import requests
from urlparse import urljoin


class Client(object):
    """Wrapper for APIs of imagej-server"""

    HOST = 'http://localhost:8080'

    def __init__(self, host=HOST, **kwargs):
        self.host = host

    def get_modules(self):
        """Gets a list of module IDs.

        API: GET /modules

        :return: a list of module IDs
        :rtype: list of string
        """

        url = urljoin(self.host, 'modules')
        r = requests.get(url)
        r.raise_for_status()
        return r.json()

    def get_module(self, id):
        """Gets the detail of one module.

        API: GET /modules/{id}

        :param id: module ID
        :return: details of module specified by ID
        :rtype: dict
        """

        url = urljoin(self.host, 'modules/%s' % id)
        r = requests.get(url)
        r.raise_for_status()
        return r.json()

    def run_module(self, id, inputs=None, process=True):
        """Runs a module with inputs.

        API: POST /modules/{id}?[process=(true,false)]

        :param id: module ID
        :param inputs: inputs for module execution
        :param process: if the execution should be pre/post processed
        :return: outputs of the execution
        :rtype: dict
        """

        url = urljoin(self.host, 'modules/%s?process=%s' %
                      (id, str(process).lower()))
        r = requests.post(url, json=inputs)
        r.raise_for_status()
        return r.json()

    def upload_file(self, data):
        """Uploads a file to imagej-server (currently only supports image files).

        API: POST /io/file

        :param data: file-like object to be uploaded
        :return: object ID representing the file in form of
                {"id": "object:1234567890abcdef"}
        :rtype: dict
        """

        url = urljoin(self.host, 'io/file')
        r = requests.post(url, files={'file': data})
        r.raise_for_status()
        return r.json()

    def request_file(self, id, format, config=None):
        """Requests to stores an object as a file in specific format for
        download.

        API: POST /io/{id}?format=FORMAT

        :param id: object ID
        :param format: format of the object to be stored in
        :param config: configuration for storing the file (not tested)
        :return: filename for downloading the requested file in the form of
                {"filename": "asdf1234.png"}
        :rtype: dict
        """

        url = urljoin(self.host, 'io/%s' % id)
        r = requests.post(
            url, params={'format': format}, json={'config': config})
        r.raise_for_status()
        return r.json()

    def retrieve_file(self, filename):
        """Retrieves the content of a file.

        API: GET /io/{filename}

        :param filename: name of file to be download
        :return: the downloaded file
        :rtype: file
        """

        url = urljoin(self.host, 'io/%s' % filename)
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    def stop(self):
        """Stops the imagej-server gracefully.

        API: DELETE /admin/stop
        """

        url = urljoin(self.host, 'admin/stop')
        r = requests.delete(url)
        r.raise_for_status()
