# Python client for imagej-server

[`imagej_server.py`](imagej_server.py) Wraps the APIs of `imagej-server` in simple Python functions. 

[`imagej_client.py`](imagej_client.py) Provides higher level functionalities that utilize the APIs.

## Requirements:

    - requests
    - Pillow

Use `pip install -r requirements.txt` to install requirements.

`Pillow` is required for the `Client.show()` function. In addition, `display` or `xv` needs to exist in your system to view the image.

## Usage:

Try running [usage.py](usage.py) in the directory of this README file and see the results.
 
```Python
# This script shows the basic functionalities of the Python imagej client.
# It uploads an image to imagej-server, inverts the image, and display both the
# original and the inverted ones.

from imagej_client import Client
import json

client = Client()

# Find modules that contain a specific string in its ID.
create = client.find("CreateImgFromImg")[0]
invert = client.find("InvertII")[0]

# Check details of a module. Names of "inputs" and "outputs" are usually important.
print('Details for CreateImgFromImg:')
print(json.dumps(client.detail(create), indent=4))
print('Details for InvertII:')
print(json.dumps(client.detail(invert), indent=4))

# Upload an image.
img_in = client.upload('../../src/test/resources/imgs/about4.tif')

# Execute modules.
result = client.run(create, {'in': img_in})
img_out = result['out']
result = client.run(invert, {'in': img_in, 'out': img_out})
img_out = result['out']

# Check objects available on imagej-server
print('Objects available on imagej-server')
print(client.objects())

# retrieve/show images.
client.retrieve(img_out, format='png', dest='/tmp')
client.show(img_in, format='tiff')
client.show(img_out, format='tiff')
```

## Documentation

The entry point of [imagej_client](imagej_client.py) is a *__Client__* object, which has the following functions:

```
class imagej_client.Client(host='http://localhost:8080')
    Creates a client that bounds to host.
       
    :param host: address of imagej-server
   
Client.detail(id)
    Gets the detail of a module specified by the ID.
       
    :param id: the ID of the module
    :return: details of a module
    :rtype: dict
   
Client.find(regex)
    Finds all module IDs that match the regular expression.
       
    :param regex: the regular express to match the module IDs
    :return: all matching IDs
    :rtype: list[string]
   
Client.modules(refresh=False)
    Gets the module IDs of imagej-server if no cache is available or
    refresh is set to True, or returns the cache for the IDs otherwise.
       
    :param refresh: force fetching modules from imagej-server if True
    :return: imagej-server module IDs
    :rtype: list[string]
   
Client.objects()
    Gets a list of objects being served on imagej-server, sorted by ID.
       
    :return: a list of object IDs
    :rtype: list[string]
   
Client.retrieve(id, format, config=None, dest=None)
    Retrieves an object in specific format from imagej-server.
       
    If dest is None, the raw content would be returned.
       
    :param id: object ID
    :param format: file format the object to be saved into
    :param config: configuration for storing the object (not tested)
    :param dest: download destination
    :return: content of the object if dest is None, otherwise None
    :rtype: string or None
   
Client.run(id, inputs=None, process=True)
    Runs a module specified by the ID with inputs.
       
    :param id: the ID of the module
    :param inputs: a dict-like object containing inputs for the execution
    :param process: if the execution should be pre/post processed
    :return: outputs of the execution
    :rtype: dict
   
Client.show(id, format, config=None)
    Retrieves and shows an object in specific format from imagej-server.
       
    :param id: object ID if format is set, or a file being served
    :param format: file format the object to be saved into
    :param config: configuration for storing the object (not tested)
   
Client.upload(filename)
    Uploads a file to imagej-server
       
    :param filename: filename of the file to be uploaded
    :return: object ID of the uploaded file
    :rtype: string
```