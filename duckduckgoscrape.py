# from duckduckgo_images_api import search

# results = search("apple")

# # results = search("ideal smile")

# print("TYPE: ", type(results))

# print([r["url"] for r in results["results"]])

from bs4 import *
import requests as rq
import os

# api-endpoint
URL = "https://duckduckgo.com/i.js"
keyword = input('Enter the search keyword : ') + " smile dental image"

# defining a params dict for the parameters to be sent to the API
PARAMS = {'l': 'us-en',
    'o': 'json',
    'q': 'book',
    'vqd': '3-160127109499719016074744569811997028386-179481262599639828155814625357171050706&f=,,,',
}

# sending get request and saving the response as response object
r = rq.get(url=URL, params=PARAMS)

# extracting data in json format
data = r.json()

img_link = data["results"][0]['image']

img_data = rq.get(img_link).content

# os.mkdir('downloads')
filename = "dataset/" + keyword + "/" + keyword + ".png"
with open(filename, 'wb+') as f:
    f.write(img_data)

print("File " + keyword + ".png successfully downloaded.")