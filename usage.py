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

# retrieve/show images.
client.retrieve(img_out, format='png', dest='/tmp')
client.show(img_in, format='tiff')
client.show(img_out, format='tiff')