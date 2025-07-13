import pandas as pd

df =pd.read_csv(r'C:\Users\Dell\Desktop\MLendtpend\data\rawdata.csv')
df.head()
df.info()

df.isnull().sum()

import os
os.makedirs(r"C:\Users\Dell\Desktop\MLendtpend\data\cleaneddata",exist_ok=True)

df.to_csv(r"C:\Users\Dell\Desktop\MLendtpend\data\cleaneddata\cleaned.csv",index=False)