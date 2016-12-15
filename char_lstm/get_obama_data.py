import os
import urllib
import zipfile
if not os.path.exists("obama_data/char_lstm.zip"):
    urllib.urlretrieve("http://data.mxnet.io/data/char_lstm.zip", "char_lstm.zip")
with zipfile.ZipFile("obama_data/char_lstm.zip","r") as f:
    f.extractall("./obama_data")     
with open('obama_data/obama.txt', 'r') as f:
    print f.read()[0:1000]
