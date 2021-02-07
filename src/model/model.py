import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

def train(data_path, data_name, test_size, model, c, save_model, load_model):
    #retrieve  model
    if(load_model = 'True'):
        mdl = joblib.load(data_path + '/' + 'trained.model')
        return mdl

    #retrieve data
    file_path = data_path + "/"
    processed_data  = pd.read_csv(file_path + dynamic_name)

    #split out the data
    train, test = train_test_split(samples, test_size=test_size, random_state = 180)

    #choose the model
    if(model = "decision tree"):
        mdl = DecisionTreeClassifier(max_depth = c)
    elif(model = "SVM"):
        mdl = SVC(C = c)
    else:
        return None

    #train model
    mtl = mtl.fit(train.drop('wait_msecs', axis = 1), train.wait_msecs)

    #report accuracy
    f = file.open(data_path + '/' + 'train_acc.txt', model = 'w')
    prediction = model.predict(train.drop('wait_msecs', axis = 1))
    f.write(metrics.classification_report(train.wait_msecs, prediction, digits=3))
    f.close()
    f = file.open(data_path + '/' + 'test_acc.txt', model = 'w')
    prediction = model.predict(test.drop('wait_msecs', axis = 1))
    f.write(metrics.classification_report(test.wait_msecs, prediction, digits=3))
    f.close()

    #save model
    if(save_model = 'True'):
        joblib.dump(mdl, data_path + '/' + 'trained.model')

    return mdl