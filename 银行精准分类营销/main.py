from sklearn.model_selection import cross_val_score, StratifiedKFold

from transfer import *
from transfer import *
from visualize import *

if __name__ == '__main__':
    # 获得数据
    train_df, test_df = load_data()

    # 数据转化
    train_df = transfer_df(train_df)

    # 分为X, y
    X = train_df.drop(columns=['y'])
    y = train_df['y']

    # StratifiedKFold
    Stratified_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    # model

    gbdt_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        # min_samples_split=100,
        max_depth=3,
        # max_features='sqrt',
        subsample=0.9,
        random_state=0
    )

    # scores = []
    # for n_estimator in range(80, 301, 10):
    #     model = GradientBoostingClassifier(n_estimators=n_estimator)
    #     model.fit(X, y)
    #     score = cross_val_score(estimator=model, X=X, y=y, cv=Stratified_kf, scoring='roc_auc', n_jobs=-1).mean()
    #     scores.append(score)
    #
    # plt.plot(range(80, 301, 10), scores, linestyle='-.')
    # plt.show()

    # validation curve

    # plot_by_param(
    #     hyperparameter='subsample',
    #     Range=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #     X=X,
    #     y=y,
    #     k=Stratified_kf,
    #     model=gbdt_model,
    #     scoring='roc_auc'
    # )

    # gridSearch

    # searcher = grid_search(
    #     model=gbdt_model,
    #     param_dict={
    #         # 'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #         'n_estimators': range(300, 501, 10),
    #     },
    #     score_method='roc_auc',
    #     k=Stratified_kf,
    #     features=X,
    #     targets=y,
    # )
    #
    # print(
    #     searcher.best_params_,
    #     searcher.best_score_)

    scores = cross_val_score(gbdt_model, X, y, cv=Stratified_kf, n_jobs=-1)
    print(scores.mean())


    gbdt_model.fit(X, y)

    # 特征重要程度
    feature_importance(X, gbdt_model)

    # 递交
    # probability = gbdt_model.predict_proba(transfer_df(test_df))
    # submit(probability[:, 1])









