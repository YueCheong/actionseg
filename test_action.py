from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR

from datasets.AS_base import SegDataset
import models.AS_p4_base as P4Models
import models.AS_pptr_base as PTTRModels


def evaluate(model1, model2, criterion, data_loader, device, len_test):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    acc_list = []

    total_correct_class = [0] * 19
    total_class = [0] * 19

    with torch.no_grad():
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        edit = 0
        length = 0
        for clip, target in metric_logger.log_every(data_loader, 20, header):
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output.permute(0,2,1), target)

            output = torch.max(output,dim=-1)[1]
            output, target = output.cpu().numpy().astype(np.int32), target.cpu().numpy().astype(np.int32)
            acc = np.mean(output == target)
            # acc = torch.mean(torch.tensor(output==target,dtype=torch.float))
            acc_list.append(acc)
            for b in range(output.shape[0]):
                # print(output[b].shape)
                # print(target[b].shape)
                edit += edit_score(output[b], target[b])
            for b in range(output.shape[0]):
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(output[b], target[b], overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1


            batch_size = clip.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    total_acc = np.mean((np.array(acc_list)))
    edit = (1.0 * edit) / len_test
    print('Edit: %.4f' % (edit))
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1
    print("total acc:", total_acc)
    return total_acc












































# # voting 投票融合：将多个模型预测结果投票，得票最多作为最终结果
# def ensemble_voting(models, X):
#     y_pred = np.zeros(X.shape[0],len(models))

#     for i, model in enumerate(models):
#         y_pred[:,i] = model.predict(X).reshape(-1)
    
#     y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=1,arr=y_pred)

#     return y_pred

# # averaging 平均融合：将多个模型的预测结果进行平均，得到最终结果。

# def ensembel_average(models, X):
#     y_pred = np.zeros(X.shape[0],len(models))

#     for i, models in enumerate(models):
#         y_pred[:,i] = model.predict(X).reshape(-1)
    
#     y_pred = np.mean(y_pred, axis=1)
#     y_pred = np.round(y_pred).astype(int)

#     return y_pred


# # stacking 堆叠融合，将多个模型的预测结果作为输入，再训练一个元模型来融合它们
# def ensemble_stacking(models, meta_model, X_train, y_train, X_test):
#     # 训练集的预测结果作为元特征
#     meta_features = np.zeros((X_train.shape[0],len(models)))
#     for i, models in enumerate(models):
#         skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
#         for train_idx, val_idx in skf.split(X_train, y_train):
#             X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
#             X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

#             model.fit(X_train_fold, y_train_fold)
#             y_pred = model.predict(X_val_fold)
#             meta_features[val_idx, i] = y_pred

#         model.fit(X_train, y_train)
    
#     # 元模型训练和预测
#     meta_model.fit(meta_features, y_train)
#     meta_features_test = np.zeros(X_test.shape[0],len(models))
#     for i, model in enumerate(models):
#         y_pred = model.predict(X_test)
#         meta_features_test[:, i] = y_pred
#     y_pred = meta_model.predict(meta_features_test)
#     y_pred = np.round(y_pred).astype(int)

#     return y_pred

    