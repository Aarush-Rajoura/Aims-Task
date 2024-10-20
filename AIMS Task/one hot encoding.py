import pandas as pd
import numpy as np


def OneHotEncoding(dataset,column):
    unique_values=sorted(dataset[column].unique())
    for values in unique_values:
        dataset[f"{values}"]=(dataset[column]==values).astype(int)
    
    dataset = dataset.drop(column, axis=1)

    return dataset

data=pd.read_csv(r"Clean_Dataset.csv")
print("before One-hot encoding encoding\n",data.head())

data=OneHotEncoding(data,"airline")

print("after One-hot encoding encoding\n",data.head())