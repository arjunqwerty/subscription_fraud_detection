import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import streamlit as st

def eval_preds(model,X,y_true,y_pred):
    # Extract task Fraud
    y_true = y_true['Fraud']
    cm = confusion_matrix(y_true, y_pred)
    # Probability of the minority class
    proba = model.predict_proba(X)[:,1]
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, proba)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    f2 = fbeta_score(y_true, y_pred, pos_label=1, beta=2)
    metrics = pd.Series(data={'ACC':acc, 'AUC':auc, 'F1':f1, 'F2':f2})
    metrics = round(metrics,3)
    return cm, metrics

def tune_and_fit(clf,X,y,params):
    f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
    # start_time = time.time()
    grid_model = GridSearchCV(clf, param_grid=params, cv=5, scoring=f2_scorer)
    grid_model.fit(X, y['Fraud'])
    # print('Best params:', grid_model.best_params_)
    # Print training times
    # train_time = time.time()-start_time
    # mins = int(train_time//60)
    # print('Training time: '+str(mins)+'m '+str(round(train_time-mins*60))+'s')
    return grid_model

def predict_and_evaluate(fitted_models,X,y_true,clf_str):
    cm_dict = {key: np.nan for key in clf_str}
    metrics = pd.DataFrame(columns=clf_str)
    y_pred = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models,clf_str):
        # Update predictions
        y_pred[model_name] = fit_model.predict(X)
        # Metrics
        cm, scores = eval_preds(fit_model,X,y_true, y_pred[model_name])
        # Update Confusion matrix and metrics
        cm_dict[model_name] = cm
        metrics[model_name] = scores
    return metrics
    # return y_pred, cm_dict, metrics

# Define a function to train and save the AI model
def train_model(csv_file, model_filename):
    st.write("Training the AI model...")
    data = pd.read_csv(csv_file)

    ## Set numeric columns dtype to float
    data['Session_Length'] = data['Session_Length'].astype('float64')

    # Drop ID columns
    df = data.copy()
    df.drop(columns=['Transaction_ID'], inplace=True)

    # Create lists of features and Fraud names
    features = [col for col in df.columns if col !='Fraud']
    num_features = [feature for feature in features if df[feature].dtype=='float64' or df[feature].dtype=='int64']

    # Scaling
    sc = StandardScaler()
    df_pre = df.copy()
    user_hist = df_pre['User_History'].unique()
    user_dict = {}
    for i in range(len(user_hist)):
        user_dict[user_hist[i]] = i
    df_pre['User_History'].replace(to_replace=user_dict, inplace=True)
    df_pre[num_features] = sc.fit_transform(df_pre[num_features])

    # train-validation-test split
    X, y = df_pre[features], df_pre[['Fraud']]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, stratify=df_pre['Fraud'], random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, stratify=y_trainval['Fraud'], random_state=0)

    # Models
    # lr = LogisticRegression(random_state=0)
    # knn = KNeighborsClassifier()
    # svc = SVC()
    rfc = RandomForestClassifier()
    # xgb = XGBClassifier()
    clf = [rfc]
    clf_str = ['RFC']

    # Parameter grids for GridSearch
    # lr_params = {'random_state':[0]}
    # knn_params = {'n_neighbors':[1,3,5,8,10]}
    # svc_params = {'C': [1, 10, 100], 'gamma': [0.1,1], 'kernel': ['rbf'], 'probability':[True], 'random_state':[0]}
    rfc_params = {'n_estimators':[100,300,500,700], 'max_depth':[5,7,10], 'random_state':[0]}
    # xgb_params = {'n_estimators':[300,500,700], 'max_depth':[5,7], 'learning_rate':[0.01,0.1], 'objective':['binary:logistic']}
    # params = pd.Series(data=[lr_params,knn_params,svc_params,rfc_params,xgb_params], index=clf)
    params = pd.Series(data=[rfc_params], index=clf)

    # Tune hyperparameters with GridSearch (estimated time 8m)
    # print('GridSearch start')
    fitted_models_binary = []
    for model, model_name in zip(clf, clf_str):
        # print('Training '+str(model_name))
        fit_model = tune_and_fit(model,X_train,y_train,params[model])
        fitted_models_binary.append(fit_model)

    # Create evaluation metrics
    metrics_val = predict_and_evaluate(fitted_models_binary,X_val,y_val,clf_str)
    metrics_test = predict_and_evaluate(fitted_models_binary,X_test,y_test,clf_str)
    metrics_final = metrics_val*metrics_test

    # Calculating best model
    # macc = 0
    # bestm = 0
    # for i, j in enumerate(clf_str):
    #     acc = metrics_final[j][0]
    #     if acc > macc:
    #         macc = acc
    #         bestm = i
    # print(bestm, "\t", clf_str[bestm])

    # Saving model to file
    best_model = fitted_models_binary[0]
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'models')
    filename = os.path.join(filename, model_filename)
    joblib.dump(best_model, filename+".pkl")

    st.write(f"Trained and saved in " + model_filename)

# Define a function to make predictions using the saved AI model
def predict(csv_file, model_filename):
    col3.write("Making predictions...")

    # Load the saved model
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'models')
    filename = os.path.join(filename, model_filename)
    model_from_file = joblib.load(filename+".pkl")
    data = pd.read_csv(csv_file)

    # Drop ID columns
    df = data.copy()
    df.drop(columns=['Transaction_ID'], inplace=True)
    features = [col for col in df.columns if col !='Fraud']
    num_features = [feature for feature in features if df[feature].dtype=='float64' or df[feature].dtype=='int64']

    # Scaling
    sc = StandardScaler()
    df_pre = df.copy()
    user_hist = df_pre['User_History'].unique()
    user_dict = {}
    for i in range(len(user_hist)):
        user_dict[user_hist[i]] = i
    df_pre['User_History'].replace(to_replace=user_dict, inplace=True)
    df_pre[num_features] = sc.fit_transform(df_pre[num_features])

    # Predict
    predictions = model_from_file.predict(df_pre[features])

    # Display Fraud and Non-Fraud rows
    col3.write("Fraud Subscriptions:")
    true_rows = data[predictions == 1]
    col3.write(true_rows['Transaction_ID'])
    # st.write("Non-Fraud Rows:")
    # false_rows = data[predictions == 0]
    # st.write(false_rows)

def list_saved_models():
    saved_models = [f[:-4] for f in os.listdir('models') if f.endswith(".pkl")]
    return saved_models

# Define the Streamlit app
st.set_page_config(page_title="AI Model App", layout="wide")

# Add a navigation bar with a logo, add a sidebar for navigation
col1, cold, col2 = st.columns([1, 3, 1])
col1.image("logo.png", width=250)

page = col2.selectbox("Select a page:", ["Train Model", "Predict"], key="unique_key")

cold1, col3, cold2 = st.columns([2, 3, 2])
col3.title("Fraud Detection App")

if page == "Train Model":
    col3.header("Train AI Model")
    uploaded_file = col3.file_uploader("Upload a CSV file for training:", type=["csv"])
    model_filename = col3.text_input("Enter the model filename (without extension):")
    if col3.button("Submit"):
        if uploaded_file is not None:
            with col3.spinner("Training in progress..."):
                train_model(uploaded_file, model_filename)
            col3.success("Training completed!")
elif page == "Predict":
    col3.header("Make Predictions")
    uploaded_file = col3.file_uploader("Upload a CSV file for predictions:", type=["csv"])
    saved_models = list_saved_models()
    if saved_models is not None:
        selected_model = col3.selectbox("Select a trained model:", saved_models)
        if col3.button("Submit"):
            if uploaded_file is not None:
                predict(uploaded_file, selected_model)
    else:
        col3.header("No models trained yet")
