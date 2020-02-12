import pandas as pd
import numpy as np
import joblib
window_size = 8
features_size = 17
def datatrans(sensors):
    for w in range(window_size-1,0,-1):
        for f in range(features_size):
            sensors[w][f] -= sensors[w-1][f]
    for f in range(features_size):
        sensors[0][f] = sensors[1][f]
    data = np.std(sensors,axis=0,ddof=1)
    data = pd.DataFrame(data).transpose()
    data.columns = ['time','gx', 'gy', 'gz', 'ax', 'ay', 'az', 'mx', 'my', 'mz', 'ox', 'oy', 'oz', 'q0', 'q1', 'q2', 'q3']
    data = data.drop('time',axis=1)
    return data
def load_models_in_disk():
    path = [
        'load_model_lgb_1',
        'load_model_lgb_0',
        'load_model_xgb_1',
        'load_model_xgb_0',
        'load_model_xgb_2'
    ]
    model_list = [joblib.load(path[i]) for i in range(len(path))]
    return model_list
def predict_proba_(data,model_list):
    data = datatrans(data)
    oof_test = np.zeros((len(data), 5), dtype=np.float)
    for index,model in enumerate(model_list):
        if index < 2:
            oof_test+=model.predict(data,num_iteration=model.best_iteration)
        else:
            oof_test+=model.predict_proba(data)
    print(oof_test)# 5个类的概率
    return list(oof_test[0])


