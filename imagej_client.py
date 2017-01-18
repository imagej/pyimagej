import requests
from urlparse import urljoin


class Client(object):
    HOST = 'http://localhost:8080'

    def __init__(self, host=HOST, **kwargs):
        self.host = host

    def get_modules(self):
        url = urljoin(self.host, 'modules')
        r = requests.get(url)
        r.raise_for_status()
        return r.json()

    def get_module(self, id):
        url = urljoin(self.host, 'modules/%s' % id)
        r = requests.get(url)
        r.raise_for_status()
        return r.json()

    def run_module(self, id, inputs=None, process=True):
        url = urljoin(self.host, 'modules/%s?process=%s' %
                      (id, str(process).lower()))
        r = requests.post(url, json=inputs)
        r.raise_for_status()
        return r.json()

    def upload_file(self, data):
        url = urljoin(self.host, 'io/file')
        r = requests.post(url, files={'file': data})
        r.raise_for_status()
        return r.json()

    def request_file(self, id, format, config=None):
        url = urljoin(self.host, 'io/%s' % id)
        r = requests.post(
            url, params={'format': format}, json={'config': config})
        r.raise_for_status()
        return r.json()

    def retrieve_file(self, filename):
        url = urljoin(self.host, 'io/%s' % filename)
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    def stop(self):
        url = urljoin(self.host, 'admin/stop')
        r = requests.delete(url)
        r.raise_for_status()
