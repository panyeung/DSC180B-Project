import pandas as pd

def get_data(data_path, dynamic_name, static_name, merge_how, merge_on):
    '''
    Retrieve and clean the data
    '''

    #retrieve data
    file_path = data_path + "/"
    dynamic = pd.read_csv(file_path + dynamic_name)
    static = pd.read_csv(file_path + static_name)
    df = dynamic.merge(static, how = merge_how, on = merge_on)

    #clean data_cfg
    df. = df.age_category.fillna(method = 'backfill').str[:1].replace({'U': '-1'})
    df

    return
