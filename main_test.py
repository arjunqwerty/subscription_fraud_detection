from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

## Import data
data_path = 'dataset/newgen1.csv'
data = pd.read_csv(data_path)

# Drop ID columns
df = data.copy()
df.drop(columns=['Transaction_ID'], inplace=True)

features = [col for col in df.columns if col !='Fraud']

num_features = [feature for feature in features if df[feature].dtype=='float64' or df[feature].dtype=='int64']

# Scaling
sc = StandardScaler()
df_pre = df.copy()
df_pre[num_features] = sc.fit_transform(df_pre[num_features])

model_from_file = joblib.load('newfile.pkl')
arr = model_from_file.predict(df_pre[features])

for i in range(len(arr)):
    if arr[i] == 1:
        print(i, end=" ")
