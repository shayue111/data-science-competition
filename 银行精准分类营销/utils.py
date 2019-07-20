import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    """ 读取数据集 """
    train_df = pd.read_csv('./datasets/train_set.csv', index_col='ID')
    test_df = pd.read_csv('./datasets/test_set.csv', index_col='ID')

    return train_df, test_df


def setting_style():
    """ 设置画图风格 """
    mpl.rcParams['font.sans-serif'] = ['SimHei']    # 设置图片可以显示汉语
    mpl.rcParams['axes.unicode_minus'] = False      # 正常显示负号
    plt.style.use('seaborn')                        # 设置绘图风格
    pd.set_option('display.max_columns', None)      # 显示所有列


def get_cate_df(df):
    """ 返回正负样本数据，结构为DataFrame"""
    df_pos = df[df['y'] == 1]
    df_neg = df[df['y'] == 0]

    return df_pos, df_neg


def submit(scores):
    """ 生成递交文件 """
    _, test_df = load_data()
    submit_csv = pd.DataFrame(data=scores, index=test_df.index, columns=['pred'])
    submit_csv.to_csv('submit.csv')
    print("File submit.csv has been created!")





