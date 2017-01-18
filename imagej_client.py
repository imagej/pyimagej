import requests
from urlparse import urljoin


class Client(object):
    HOST = 'http://localhost:8080'

    def __init__(self, host=HOST, **kwargs):
        self.host = host

    def getModules(self):
        url = urljoin(self.host, 'modules')
        r = requests.get(url)
        r.raise_for_status()
        return r.json()

    def getWidget(self, id):
        url = urljoin(self.host, 'modules/%s' % id)
        r = requests.get(url)
        r.raise_for_status()
        return r.json()

    def runModule(self, id, inputs=None, process=True):
        url = urljoin(self.host, 'modules/%s' % id)
        runSpec = {"inputs": inputs, "process": process}
        r = requests.post(url, json=runSpec)
        r.raise_for_status()
        return r.json()

    def uploadFile(self, data):
        url = urljoin(self.host, 'io/file')
        r = requests.post(url, files={'file': data})
        r.raise_for_status()
        return r.json()

    def requestFile(self, id, format, config=None):
        url = urljoin(self.host, 'io/%s' % id)
        r = requests.post(
            url, params={'format': format}, json={'config': config})
        r.raise_for_status()
        return r.json()

    def retrieveFile(self, filename):
        url = urljoin(self.host, 'io/%s' % filename)
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    def stop(self):
        url = urljoin(self.host, 'admin/stop')
        r = requests.delete(url)
        r.raise_for_status()
