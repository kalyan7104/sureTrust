import pandas as pd

import numpy as np

d=pd.read_csv("/home/kalyan/Downloads/Titanic-Dataset.csv")

df=pd.DataFrame(d)
print(df)

data = df[['Age', 'Fare', 'Survived']].to_numpy() #to_numpy() is used for coverting data frames into numpy arrays
print(data)


clean_data = data[~np.isnan(data).any(axis=1)]
#This line removes rows containing any NaN values from a NumPy array `data` axis=1 specifiesrow 
#~ inverts to keep only rows. 

age_mean, age_std = np.mean(clean_data[:, 0]), np.std(clean_data[:, 0]) 
#calculates the mean (age_mean) and standard deviation (age_std) of the first column (index 0) in the clean_data array, which typically represents age values in the dataset.
fare_mean, fare_std = np.mean(clean_data[:, 1]), np.std(clean_data[:, 1]) 
#calculates the mean (`fare_mean`) and standard deviation (`fare_std`) of the second column (index 1) in `clean_data`, which contains fare values.
clean_data[:, 0] = (clean_data[:, 0] - age_mean) / age_std  
clean_data[:, 1] = (clean_data[:, 1] - fare_mean) / fare_std  


survived = clean_data[clean_data[:, 2] == 1] # filters `clean_data` to create a new array `survived` containing only rows where the third column (index 2, survival status) equals 1 (indicating passengers who survived).

#filters `clean_data` to create a new array `not_survived` containing only rows where the third column (index 2, survival status) equals 0 .
not_survived = clean_data[clean_data[:, 2] == 0]

mean_age_survived = np.mean(survived[:, 0]) #calculates the average age (`mean_age_survived`) from the first column (index 0) of the `survived` array containing passengers who survived.
mean_fare_survived = np.mean(survived[:, 1])

mean_age_not_survived = np.mean(not_survived[:, 0])
mean_fare_not_survived = np.mean(not_survived[:, 1])

print("Survivors - Mean age:", mean_age_survived, "Mean fare:", mean_fare_survived)
print("Non-survivors - Mean age:", mean_age_not_survived, "Mean fare:", mean_fare_not_survived)


original_fares = data[~np.isnan(data).any(axis=1)][:, 1]  # Get clean fares


fare_mean = np.mean(original_fares) #calucates_mean
fare_classification = np.where(original_fares < fare_mean, "Low", "High") #calssification of fares is performed which are less than fare_mean

fare_class_numeric = np.where(original_fares < fare_mean, 0, 1)
clean_data = np.column_stack((clean_data, fare_class_numeric))





data2=pd.read_excel("/home/kalyan/Downloads/iris.xlsx")

pip install openpyxl==3.1.0

data2=pd.read_excel("/home/kalyan/Downloads/iris.xlsx") #loads the  excel data

iris=pd.DataFrame(data2) # creates dataframe

print(iris)   #prints the dataframe

# 2. Create ratio
iris['petal_ratio'] = iris['petal.length'] / iris['petal.width']

# 3a. Average ratio by species
print("Average petal ratio by species:")
print(iris.groupby('variety')['petal_ratio'].mean())

# 3b. Highest sepal length std
print("\nSpecies with highest sepal length std:", 
      iris.groupby('variety')['sepal.length'].std().idxmax())

# 3c. Filter wide sepals
wide_sepals = iris[iris['sepal.width'] > iris['sepal.width'].mean()]
print("\nRows with above-mean sepal width:", len(wide_sepals))

# 3d. Split and combine
combined = pd.concat([
    iris[iris['petal_ratio'] < 2],
    iris[iris['petal_ratio'] >= 2]
], axis=0)
print("\nCombined DataFrame shape:", combined.shape)


