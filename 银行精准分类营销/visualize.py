from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve

from utils import *


def plot_by_param(hyperparameter, Range, X, y, model, k=5, scoring='accuracy'):
    """
    这个可视化函数展现随着某个超参数的变化，模型的在训练集和测试集的性能变化曲线

    :param hyperparameter: 超参数，string
    :param Range: 超参数变化范围，range
    :param X: np.arrays
    :param y: np.arrays
    :param model: 模型
    :param k: 交叉验证折数
    :param scoring: 性能指标
    :return: None
    """
    # Calculate accuracy on training and test set using range of parameter values
    setting_style()
    train_scores, test_scores = validation_curve(
        model,
        X,
        y,
        param_name=hyperparameter,
        param_range=Range,
        cv=k,
        scoring=scoring,
        n_jobs=-1
    )

    # print("train_scores:\n", train_scores.shape, '\n', "test_scores:\n", test_scores.shape)

    # 计算超参数取不同值时对于训练集得到的k个误差的均值和方差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # 计算超参数取不同值时对于测试集得到的k个误差的均值和方差
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 画出曲线
    plt.plot(Range, train_mean, linestyle='-', label="Training score", color="lawngreen")
    plt.plot(Range, test_mean, linestyle='--', label="Cross-validation score", color="gold")

    # 画出边界
    plt.fill_between(Range, train_mean - train_std, train_mean + train_std, color="lightsteelblue")
    plt.fill_between(Range, test_mean - test_std, test_mean + test_std, color="darkcyan")

    # 生成
    plt.title("Validation Curve")
    plt.xlabel(f"number of {hyperparameter}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()


def visual_confusion_matrix(model, features_train, target_train, features_test, target_test):
    setting_style()
    # Train model and make predictions
    target_predicted = model.fit(features_train, target_train).predict(features_test)

    # Create confusion matrix
    matrix = confusion_matrix(target_test, target_predicted)

    # Create pandas dataframe
    class_names = np.unique(target_train)
    dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()


def feature_importance(features, model):
    predictors = list(features)
    feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Importance of Features')
    plt.xticks(rotation=30)
    plt.ylabel('Feature Importance Score')
    plt.show()
