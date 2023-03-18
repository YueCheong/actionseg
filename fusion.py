from skleaen.model_selection import StratifiedKFold
import numpy as np


# voting 投票融合：将多个模型预测结果投票，得票最多作为最终结果
def ensemble_voting(models, X):
    y_pred = np.zeros(X.shape[0],len(models))

    for i, model in enumerate(models):
        y_pred[:,i] = model.predict(X).reshape(-1)
    
    y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=1,arr=y_pred)

    return y_pred

# averaging 平均融合：将多个模型的预测结果进行平均，得到最终结果。

def ensembel_average(models, X):
    y_pred = np.zeros(X.shape[0],len(models))

    for i, models in enumerate(models):
        y_pred[:,i] = model.predict(X).reshape(-1)
    
    y_pred = np.mean(y_pred, axis=1)
    y_pred = np.round(y_pred).astype(int)

    return y_pred


# stacking 堆叠融合，将多个模型的预测结果作为输入，再训练一个元模型来融合它们
def ensemble_stacking(models, meta_model, X_train, y_train, X_test):
    # 训练集的预测结果作为元特征
    meta_features = np.zeros((X_train.shape[0],len(models)))
    for i, models in enumerate(models):
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            meta_features[val_idx, i] = y_pred

        model.fit(X_train, y_train)
    
    # 元模型训练和预测
    meta_model.fit(meta_features, y_train)
    meta_features_test = np.zeros(X_test.shape[0],len(models))
    for i, model in enumerate(models):
        y_pred = model.predict(X_test)
        meta_features_test[:, i] = y_pred
    y_pred = meta_model.predict(meta_features_test)
    y_pred = np.round(y_pred).astype(int)

    return y_pred

    