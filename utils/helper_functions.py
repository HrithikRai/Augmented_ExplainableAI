import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

class Loader:

    def __init__(self, file_path):
        """Initialize with the path to the data file."""
        self.data = None
        self.file_path = file_path
        self.scaler = StandardScaler()

    def load_data(self):
        """Load Data"""
        self.data = pd.read_csv(self.file_path)

    def get_column_types(self):
        """
            Returns two lists:
            - categorical_columns: columns with object or category dtype
            - numerical_columns: columns with int or float dtype
        """
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = self.data.select_dtypes(include=['number']).columns.tolist()
        print("cat cols",categorical_columns)
        print(numerical_columns)
        return categorical_columns, numerical_columns
    
    def preprocess(self):
        """Preprocess Data"""
        categorical_columns, numerical_columns = self.get_column_types()
        # One hot encode categorical columns
        try:
            encoded = pd.get_dummies(self.data[categorical_columns],prefix=categorical_columns)
            self.data = pd.concat([encoded,self.data], axis=1).drop(categorical_columns, axis=1)
        except: pass
        # Data Imputation

        try:
            for col in numerical_columns[:-1]:
                if self.data[col].isnull().any():
                    mean_value = self.data[col].mean()
                    self.data[col].fillna(mean_value, inplace=True)
            self.data[numerical_columns[:-1]] = self.scaler.fit_transform(self.data[numerical_columns[:-1]])
        except: pass

        # Scale
        
    def get_data_split(self):
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            return train_test_split(X, y, test_size=0.20)
    
    def oversample(self, X_train, y_train):
            oversample = RandomOverSampler(sampling_strategy='minority')
            # Convert to numpy and oversample
            x_np = X_train.to_numpy()
            y_np = y_train.to_numpy()
            x_np, y_np = oversample.fit_resample(x_np, y_np)
            # Convert back to pandas
            x_over = pd.DataFrame(x_np, columns=X_train.columns)
            y_over = pd.Series(y_np, name=y_train.name)
            return x_over, y_over
    


        
        
            