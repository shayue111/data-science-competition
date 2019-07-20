"""测试一些功能的可用性"""

from utils import *


if __name__ == '__main__':
    train_data, _ = load_data()
    # print(train_data['y'].value_counts(sort=True, normalize=True, ascending=True))
    print(0.883043 / 0.116957)