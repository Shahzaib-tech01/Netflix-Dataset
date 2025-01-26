# Netflix Data Analysis

```python
print("Running")
```

## Data Loading

```python
import pandas as pd
pd.read_csv("Netflix.csv")
```

```python
df=pd.read_csv("Netflix.csv")
```

```python
df.shape
```

```python
df.head(3)
```

```python
df.tail(3)
```

```python
df.sample(3)
```

```python
df.columns
```

```python
df.rename(columns={'Age': 'Aged'}, inplace=True)
```

```python
df.describe()
```

```python
df.describe(include="all")
```

```python
df.info()
```

```python
df.isnull().sum()
```

## Data Cleaning

```python
df.dropna(inplace=True)
```

```python
df.dropna(axis=1, inplace=True)  
```

```python
df.fillna(value=99, inplace=True)
```

```python
df.drop_duplicates(inplace=True)
```

```python
df.loc[1, "Age"]=999
```

```python
df[['Age','User ID']].corr()
```

```python
df['col'] = 12
```

```python
df.drop(['col','Age'], axis=1,inplace=True)
```

```python
df.drop(1,inplace=True)
```

```python
df[3:5]
```

```python
df[2:5:9]
```

## Data PreProcessing

```python
def load_data(file_path):
    return pd.read_csv(file_path)
```

```python
def handle_missing_values(df, strategy="mean"):
    if strategy == "mean":
        return df.fillna(df.mean())
    elif strategy == "median":
        return df.fillna(df.median())
    elif strategy == "mode":
        for column in df.select_dtypes(include=['object', 'category']):
            df[column].fillna(df[column].mode()[0], inplace=True)
        return df
    else:
        raise ValueError("Unsupported strategy. Use 'mean', 'median', or 'mode'.")
```

```python
def encode_categorical(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)
```

```python
def scale_features(df):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
    return df
```

```python
def preprocess_data(df, target_column, missing_value_strategy="mean"):
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_value_strategy)

    # Encode categorical variables
    df = encode_categorical(df)

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Feature scaling
    X = scale_features(X)

    return X, y
```

```python
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
```

## Data Agumentation

```python
df.sample(frac=1).reset_index(drop=True)
```

```python
pd.get_dummies(df, columns=['Gender'], drop_first=True)
```

```python
bootstrap_sample = df.sample(n=len(df), replace=True)
```

## Data Ploting

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
df["Device"].value_counts().plot(kind='bar')
```

```python
df["Device"].value_counts().plot(kind='barh')
```

```python
df["Device"].value_counts().plot(kind='box')
```

```python
df["Device"].value_counts().plot(kind='area')
```

```python
df["Device"].value_counts().plot(kind='density')
```

```python
df["Device"].value_counts().plot(kind='line')
```

```python
df["Device"].value_counts().plot(kind='pie')
```

## Data Save

```python

df.to_excel('data.xlsx', index=False)
print("Data saved to 'data.xlsx'")
```

