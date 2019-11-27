import requests
import sys

with open(sys.argv[1], 'rb') as f:
    r = requests.post("http://0.0.0.0:5000/api/v1/predictGesture",
                        files={'file':f})

    print r.text
