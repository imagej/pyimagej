# Python client for imagej-server

[`imagej.py`](imagej.py) provides a high-level entry point `imagej.IJ` for invoking `imagej-server` APIs, which have been wrapped into simple Python functions.

## Requirements:

    - requests
    - Pillow

Use `pip install -r requirements.txt` to install requirements.

`Pillow` is required for the `IJ.show()` function. In addition, `display` or `xv` needs to exist in your system to view the image.

## Usage:

Try running [usage.py](usage.py) in the directory of this README file and see the results.
 
```Python
# This script shows the basic functionalities of the Python imagej client.
# It uploads an image to imagej-server, inverts the image, and display both the
# original and the inverted ones.

import imagej
import json

ij = imagej.IJ()

# Find modules that contain a specific string in its ID.
create = ij.find("CreateImgFromImg")[0]
invert = ij.find("InvertII")[0]

# Check details of a module. Names of "inputs" and "outputs" are usually important.
print('Details for CreateImgFromImg:')
print(json.dumps(ij.detail(create), indent=4))
print('Details for InvertII:')
print(json.dumps(ij.detail(invert), indent=4))

# Upload an image.
img_in = ij.upload('../../src/test/resources/imgs/about4.tif')

# Execute modules.
result = ij.run(create, {'in': img_in})
img_out = result['out']
result = ij.run(invert, {'in': img_in, 'out': img_out})
img_out = result['out']

# retrieve/show images.
ij.retrieve(img_out, format='png', dest='/tmp')
ij.show(img_in, format='jpg')
ij.show(img_out)

```

## Documentation

The entry point of imagej.py is an *__IJ__* object, which has the following functions:

```
class imagej.IJ(host='http://localhost:8080')
    Creates a client that bounds to host.
       
    :param host: address of imagej-server
   
IJ.detail(id)
    Gets the detail of a module or an object specified by the ID.

    :param id: the ID of a module or an object
    :return: details of a module or an object
    :rtype: dict
   
IJ.find(regex)
    Finds all module IDs that match the regular expression.
       
    :param regex: the regular express to match the module IDs
    :return: all matching IDs
    :rtype: list[string]
   
IJ.modules(refresh=False)
    Gets the module IDs of imagej-server if no cache is available or
    refresh is set to True, or returns the cache for the IDs otherwise.
       
    :param refresh: force fetching modules from imagej-server if True
    :return: imagej-server module IDs
    :rtype: list[string]
   
IJ.objects()
    Gets a list of objects being served on imagej-server, sorted by ID.
       
    :return: a list of object IDs
    :rtype: list[string]
    
IJ.remove(id)
    Removes one object from imagej-server.
        
    :param id: object ID to remove
   
IJ.retrieve(id, format, config=None, dest=None)
    Retrieves an object in specific format from imagej-server.
       
    If dest is None, the raw content would be returned.
       
    :param id: object ID
    :param format: file format the object to be saved into
    :param config: configuration for storing the object (not tested)
    :param dest: download destination
    :return: content of the object if dest is None, otherwise None
    :rtype: string or None
   
IJ.run(id, inputs=None, process=True)
    Runs a module specified by the ID with inputs.
       
    :param id: the ID of the module
    :param inputs: a dict-like object containing inputs for the execution
    :param process: if the execution should be pre/post processed
    :return: outputs of the execution
    :rtype: dict
   
IJ.show(id, format, config=None)
    Retrieves and shows an object in specific format from imagej-server.
       
    :param id: object ID if format is set, or a file being served
    :param format: file format the object to be saved into
    :param config: configuration for storing the object (not tested)
   
IJ.upload(filename)
    Uploads a file to imagej-server
       
    :param filename: filename of the file to be uploaded
    :param type: optional hint for file type
    :return: object ID of the uploaded file
    :rtype: string
```