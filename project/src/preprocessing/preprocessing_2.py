import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Define encoder path
encoder_path = {
    "buying" : "../../model/ohe_buying.pkl",
    "maint" : "../../model/ohe_maint.pkl",
    "doors" : "../../model/ohe_doors.pkl",    
    "person" : "../../model/ohe_person.pkl",    
    "lug_boot" : "../../model/ohe_lug_boot.pkl",    
    "safety" : "../../model/ohe_safety.pkl",
    "target" : "../../model/le_target.pkl"
}

# Define dataset path
dataset_path = "../../data/processed/car_data.csv"
dataset_dest_path = "../../data/processed/car_dataset.pkl"

# Define dataset separator and index
sep = "\t"
index_col = "index"

# Define list of columns
list_columns = ['buying', 'maint', 'doors', 'person', 'lug_boot', 'safety', 'target']

# Define column categories to encode
column_categories = {
    "buying" : np.array(['vhigh', 'high', 'med', 'low']).reshape(-1, 1),
    "maint" : np.array(['vhigh', 'high', 'med', 'low']).reshape(-1, 1),
    "doors" : np.array(['2', '3', '4', '5more']).reshape(-1, 1),
    "person" : np.array(['2', '4', 'more']).reshape(-1, 1),
    "lug_boot" : np.array(['small', 'med', 'big']).reshape(-1, 1),
    "safety" : np.array(['low', 'med', 'high']).reshape(-1, 1),
    "target" : np.array(['unacc', 'acc', 'vgood', 'good']).reshape(-1, 1)
}

def main():
    
    # Read dataset
    data = pd.read_csv(dataset_path, sep= "\t", index_col= "index")
    
    # Rename column in dataset
    data.columns = list_columns
    
    # Remove "index" column in dataset
    data.index.name = None
    
    for column in data.columns:
        if (column != "target"):
            ohe = OneHotEncoder(sparse_output= False)
            ohe.fit(column_categories[column])
            temp = pd.DataFrame(
                ohe.transform(data[column].to_numpy().reshape(-1, 1)),
                columns= [column + "_" + name for name in ohe.categories_[0].tolist()]
            )
            data = pd.concat([data, temp], axis= 1)
            data.drop(columns= column, inplace= True)
            joblib.dump(ohe, encoder_path[column])
            print(f'One Hot Encoding data {column} completed!')
            
        elif(column == "target"):
            le = LabelEncoder()
            le.fit(column_categories[column].ravel())
            data[column] = le.transform(data[column])
            joblib.dump(le, encoder_path[column])
            print(f'Label encoding data {column} success!')
            
    data.to_pickle(dataset_dest_path)
    
if __name__ == "__main__":
    main()