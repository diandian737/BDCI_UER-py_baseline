#!/usr/bin/env python
# coding: utf-8
import lightgbm as lgb
import numpy as np
import json
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import warnings
import os

# 根据模型的数量加载特征，特征以 model-i 命名，与run_classifier_cv.py 脚本中命名一致
# data_path 是UER-py训练集路径
model_num = 9

def load_features(feature_path, data_path):

    features = []
    for i in range(0,model_num):
        train_x = np.load(feature_path + 'model-'+str(i)+'.npy')
        features.append(train_x)
    features = np.concatenate(features, axis=-1)

    if data_path:
        labels = []
        with open(data_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            labels.append(int(line.split('\t')[0]))

        return features, labels
    else:
        return features



# 加载训练集、测试集特征
train_data, train_labels = load_features('features/train/', 'ccf_total.tsv')
test_data = load_features('features/test/', None)



# 设置 LGB 超参数，可以根据贝叶斯优化结果调整
params = {
    'task': 'train',
    'boosting_type': 'dart',  # 设置提升类型
    #'objective': 'multiclassova', # 目标函数
    'objective': 'multiclass',
    'num_class': 2,  
    'metric': 'multi_error',
    #'learning_rate': 0.05,  # 学习速率

    'save_binary': True,
    'max_bin': 63,
    'bagging_fraction': 0.4,
    'bagging_freq': 5,

    'feature_fraction': 0.03901,
    'lambda_l1':  7.039,
    'lambda_l2': 0.7008 ,
    'learning_rate': 0.157,
    'max_depth': 68,
    'min_data_in_leaf': 34,
    'min_gain_to_split':0.4377,
    'min_sum_hessian_in_leaf':15,
    'num_leaves': 72
}


#划分训练、验证集，训练LGB模型
x_train,x_val,y_train,y_val=train_test_split(train_data,train_labels,test_size=0.1)

lgb_train = lgb.Dataset(x_train, y_train) 
lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)  


model = lgb.train(params,lgb_train,num_boost_round=int(params['rounds']),valid_sets=lgb_eval, early_stopping_rounds=130)

pred = model.predict(x_val)
val_pred = np.argmax(pred,axis=1)


confusion = np.zeros((2,2))

for i in range(len(pred)):
    confusion[val_pred[i],y_val[i]] += 1
correct = np.sum(val_pred == y_val)

marco_f1 = []
for i in range(2):
    try:
        p = confusion[i,i].item()/confusion[i,:].sum().item()
        r = confusion[i,i].item()/confusion[:,i].sum().item()
        f1 = 2*p*r / (p+r)
    except:
        f1 = 0
        
    print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
    marco_f1.append(f1)

print("Macro F1: {:.4f}".format(np.mean(marco_f1)))
print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(val_pred), correct, len(val_pred)))




# 十折训练并保存模型
model_name = '9_models'
os.system('mkdir models/'+model_name)
fold_num = 10

labels=train_labels
train_x = train_data
test_x = test_data
score = []

per_flod_num = len(labels) // 11

pred = np.zeros([test_data.shape[0],2])
for fold in range(fold_num):
    x_train = np.concatenate((train_x[0:fold * per_flod_num], train_x[(fold+1)*per_flod_num:]),axis = 0)
    x_val = train_x[fold * per_flod_num: (fold+1)*per_flod_num]
    y_train = labels[0:fold * per_flod_num] + labels[(fold+1)*per_flod_num:]
    y_val = labels[fold * per_flod_num: (fold+1)*per_flod_num]


    lgb_train = lgb.Dataset(x_train, y_train) 
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)  

    gbm = lgb.train(params,lgb_train,num_boost_round=int(params['rounds']),valid_sets=lgb_eval, verbose_eval=0)

    val_pred = gbm.predict(x_val)

    val_pred = np.argmax(val_pred,axis=1)

    confusion = np.zeros((2,2))

    for i in range(len(val_pred)):
        confusion[val_pred[i],y_val[i]] += 1
    correct = np.sum(val_pred == y_val)

    marco_f1 = []
    for i in range(2):
        p = confusion[i,i].item()/confusion[i,:].sum().item()
        r = confusion[i,i].item()/confusion[:,i].sum().item()
        f1 = 2*p*r / (p+r)
        marco_f1.append(f1)
        print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
    score.append(np.mean(marco_f1))
    #print("Macro F1: {:.4f}".format(np.mean(marco_f1)))
    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(val_pred), correct, len(val_pred)))
    gbm.save_model('models/'+model_name+'/fold'+str(fold)+'.lgb')
    pred += gbm.predict(test_x)

pred = np.argmax(pred,axis=1)

# 将推理结果转换格式，形成提交文件，'sample_submission.tsv' 是比赛官网下载的示例

with open('sample_submission.tsv') as f:
    sample = f.readlines()

    
with open('submission_ensemble.tsv', 'w') as f:
    for line,ans in zip(sample,pred):
        q_id, a_id, _ = line.strip().split('\t')
        f.write(q_id)
        f.write('\t')
        f.write(a_id)
        f.write('\t')
        f.write(str(ans))
        f.write('\n')
    
