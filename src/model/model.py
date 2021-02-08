import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import metrics
import joblib


def train(data_path, data_name, test_size, model, c, save_model, load_model):
    #retrieve  model
    if(load_model == 'True'):
        mdl = joblib.load(data_path + '/' + 'trained.model')
        return mdl

    #retrieve data
    file_path = data_path + "/"
    processed_data  = pd.read_csv(file_path + data_name)

    #pipelines
    processed_data = processed_data.drop(['batch_id', 'guid', 'wait_msecs'], axis = 1)
    nominal = ['chassistype', 'os', 'graphicscardclass', 'cpucode', 'persona']
    prepca = processed_data[nominal].fillna(method = 'backfill')
    pipe = OneHotEncoder()
    one_hot_system = pipe.fit(prepca).transform(prepca).todense()
    quan = ['before_cpuutil_max', 'before_harddpf_max', 'before_diskutil_max', 'ram', '#ofcores', 'age_category', 'processornumber']
    quan_features = processed_data[quan].values
    features = np.append(one_hot_system, quan_features, axis = 1)
    pca=PCA(n_components=30)
    cols = pca.fit(features).transform(features)
    processed_df = processed_data[['target']]
    index = 1
    for i in cols.transpose():
        processed_df['feature_' + str(index)] =  pd.Series(i, index = processed_data.index)
        index += 1


    #split out the data
    train, test = train_test_split(processed_df, test_size= float(test_size), random_state = 180)

    #choose the model
    mdl = None
    if(model == "decision tree"):
        mdl = DecisionTreeClassifier(max_depth = int(c))
    elif(model == "SVM"):
        mdl = SVC(C = int(c))
    else:
        return None

    #train model
    mdl = mdl.fit(train.drop('target', axis = 1), train.target)

    #report accuracy
    f = open(data_path + '/' + 'train_acc.txt', mode = 'w')
    prediction = mdl.predict(train.drop('target', axis = 1))
    print("---------------------------------------------------")
    print("Accuracy on Train Set:")
    print(metrics.classification_report(train.target, prediction, digits=3))
    f.write(metrics.classification_report(train.target, prediction, digits=3))
    f.close()
    f = open(data_path + '/' + 'test_acc.txt', mode = 'w')
    prediction = mdl.predict(test.drop('target', axis = 1))
    print("---------------------------------------------------")
    print("Accuracy on Test Set:")
    print(metrics.classification_report(test.target, prediction, digits=3))
    f.write(metrics.classification_report(test.target, prediction, digits=3))
    f.close()

    #save model
    if(save_model == 'True'):
        joblib.dump(mdl, data_path + '/' + 'trained.model')

    return mdl