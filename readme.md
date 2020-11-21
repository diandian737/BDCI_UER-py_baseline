## 房产行业聊天问答匹配 baseline

比赛报名及数据集下载：https://www.datafountain.cn/competitions/474/datasets

预训练模型框架：https://github.com/dbiir/UER-py/

`ccf_total.tsv` 和 `ccf_test.tsv` 为转换成 UER-py 格式后的比赛训练集和测试集。

## 流程

首先从[https://github.com/dbiir/UER-py/wiki/Modelzoo](https://github.com/dbiir/UER-py/wiki/Modelzoo)下载一些开源预训练模型，利用 `run_classifier_cv.py`提取特征：
```
python run_classifier_cv.py --pretrained_model_path models/mixed_corpus_bert_large_model_uer.bin \
    --vocab_path models/google_zh_vocab.txt \
    --output_model_path models/ccf_beike_model_1.bin \
    --config_path models/bert_large_config.json \
    --train_path datasets/ccf_beike/train.tsv \
    --train_features_path datasets/ccf_beike/features/train/model-1.npy \
    --folds_num 5 --epochs_num 3 --batch_size 48 --encoder bert

python3 inference/run_classifier_infer_cv.py --load_model_path models/ccf_beike_model_1.bin \
    --vocab_path models/google_zh_vocab.txt \
    --config_path models/bert_large_config.json \
    --test_path datasets/ccf_beike/test.tsv \
    --test_features_path datasets/ccf_baike/features/test/model-1.npy \
    --folds_num 5 --labels_num 2 --encoder bert
```

将训练集、测试集特征分别放在 `features/train/` 和 `features/test/` 目录

修改 `lgb.py` 中13行 model_num 为使用的模型数量并运行，获得可以直接提交的 submission_ensemble.tsv 文件

------------
coggle
