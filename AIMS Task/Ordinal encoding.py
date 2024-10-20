import pandas as pd
import numpy as np
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")

print("Path to dataset files:", path)

def ordinal_encoding(dataset,column):
    
    unique_values=sorted(dataset[column].unique())
    value_to_number_dict = {value: i for i, value in enumerate(unique_values)}

    dataset[column] = dataset[column].map(value_to_number_dict)
    return dataset


data=pd.read_csv(r"Clean_Dataset.csv")
print("before ordinal encoding\n",data.head())

object_cols=[]
for i in data.columns:
    if data[i].dtype=="object":
        object_cols.append(i)
# print(object_cols)

for j in object_cols:
    ordinal_encoding(data,j)

print("after ordianl encoding\n",data.head())