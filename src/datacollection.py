import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

print("Path to dataset files:", path)

import os
dataset = r'C:\Users\Dell\Desktop\MLendtpend\data'
os.makedirs(dataset,exist_ok=True)

import pandas as pd


df= pd.read_csv(r'C:\Users\Dell\.cache\kagglehub\datasets\uciml\default-of-credit-card-clients-dataset\versions\1\UCI_Credit_Card.csv')

df.head()

os.makedirs(r'C:\Users\Dell\Desktop\MLendtpend\data\rawdata',exist_ok= True)

df.to_csv(r'C:\Users\Dell\Desktop\MLendtpend\data\rawdata.csv',index=False)

