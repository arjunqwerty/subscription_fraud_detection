import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import time

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
    return y_pred, cm_dict, metrics

def tune_and_fit(clf,X,y,params):
    f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
    start_time = time.time()
    grid_model = GridSearchCV(clf, param_grid=params, cv=5, scoring=f2_scorer)
    grid_model.fit(X, y['Fraud'])
    # print('Best params:', grid_model.best_params_)
    # Print training times
    train_time = time.time()-start_time
    mins = int(train_time//60)
    print('Training time: '+str(mins)+'m '+str(round(train_time-mins*60))+'s')
    return grid_model

## Import data
data_path = 'dataset/newgen2.csv'
data = pd.read_csv(data_path)
n = data.shape[0]

## Set numeric columns dtype to float
data['Session_Length'] = data['Session_Length'].astype('float64')

# Drop ID columns
df = data.copy()
df.drop(columns=['Transaction_ID'], inplace=True)

# Create lists of features and Fraud names
features = [col for col in df.columns if col !='Fraud']
Fraud = ['Fraud']

num_features = [feature for feature in features if df[feature].dtype=='float64' or df[feature].dtype=='int64']

# Scaling
sc = StandardScaler()
df_pre = df.copy()
user_hist = df_pre['User_History'].unique()
user_dict = {}
for i in range(len(user_hist)):
    user_dict[user_hist] = i
df_pre['User_History'].replace(to_replace=user_dict, inplace=True)
df_pre[num_features] = sc.fit_transform(df_pre[num_features])

# train-validation-test split
X, y = df_pre[features], df_pre[['Fraud']]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, stratify=df_pre['Fraud'], random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, stratify=y_trainval['Fraud'], random_state=0)

cm_labels = ['Not Fraud', 'Fraud']

# Models
lr = LogisticRegression(random_state=0)
knn = KNeighborsClassifier()
svc = SVC()
rfc = RandomForestClassifier()
xgb = XGBClassifier()
clf = [lr, knn,svc,rfc,xgb]
clf_str = ['LR','KNN','SVC','RFC','XGB']

# Parameter grids for GridSearch
lr_params = {'random_state':[0]}
knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100], 'gamma': [0.1,1], 'kernel': ['rbf'], 'probability':[True], 'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700], 'max_depth':[5,7,10], 'random_state':[0]}
xgb_params = {'n_estimators':[300,500,700], 'max_depth':[5,7], 'learning_rate':[0.01,0.1], 'objective':['binary:logistic']}
params = pd.Series(data=[lr_params,knn_params,svc_params,rfc_params,xgb_params], index=clf)

# Tune hyperparameters with GridSearch (estimated time 8m)
print('GridSearch start')
fitted_models_binary = []
for model, model_name in zip(clf, clf_str):
    print('Training '+str(model_name))
    fit_model = tune_and_fit(model,X_train,y_train,params[model])
    fitted_models_binary.append(fit_model)

# Create evaluation metrics
y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(fitted_models_binary,X_val,y_val,clf_str)
y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(fitted_models_binary,X_test,y_test,clf_str)
metrics_final = metrics_val*metrics_test

# # Show Validation Confusion Matrices
# fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(15,8))
# fig.suptitle('Validation and Test Set Confusion Matrices')
# for j, model_name in enumerate(clf_str):
#     ax = axs[0,j]
#     sns.heatmap(ax=ax, data=cm_dict_val[model_name], annot=True, fmt='d', cmap='Blues', cbar=False)
#     ax.title.set_text(model_name)
#     ax.set_xticklabels(cm_labels)
#     ax.set_yticklabels(cm_labels)
# for j, model_name in enumerate(clf_str):
#     ax = axs[1,j]
#     sns.heatmap(ax=ax, data=cm_dict_test[model_name], annot=True, fmt='d', cmap='Blues', cbar=False)
#     ax.title.set_text(model_name)
#     ax.set_xticklabels(cm_labels)
#     ax.set_yticklabels(cm_labels)
# # plt.show()
# plt.savefig("confusion_matrix.jpg")

# Print scores
print('Validation scores:', metrics_val, sep='\n')
print('Test scores:', metrics_test, sep='\n')
print('Final scores:', metrics_final, sep='\n')

# Calculating best model
macc = 0
bestm = 0
for i, j in enumerate(clf_str):
    acc = metrics_final[j][0]
    if acc > macc:
        macc = acc
        bestm = i
print(bestm, "\t", clf_str[bestm])

# Saving model to file
best_model = fitted_models_binary[bestm]
joblib.dump(best_model, 'newfile.pkl')
