from sklearn.model_selection import GridSearchCV


def grid_search(model, param_dict, score_method, k, features, targets):
    """
    这个函数用于对param_dict的参数调参，返回一个GridSearchCV类型的模型
    :param model: 模型
    :param param_dict: 参数，字典形式，键是待调整的参数名字，值是range类型
    :param score_method: 性能评价指标
    :param k: 交叉验证折树
    :param features: 训练集特征
    :param targets: 训练集标签
    :return: 训练好的GridSearchCV类型的模型
    """
    grid_searcher = GridSearchCV(
        estimator=model,
        param_grid=param_dict,
        scoring=score_method,
        n_jobs=-1,
        iid=False,
        cv=k
    )
    grid_searcher.fit(features, targets)
    return grid_searcher
