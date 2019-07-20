def job_transfer(x):
    """实现工作属性到数值的转换"""

    # """第一种"""
    # jobs = ['management', 'technician', 'admin.', 'services', 'retired', 'student',
    #         'blue-collar', 'unknown', 'entrepreneur', 'housemaid', 'self-employed',
    #         'unemployed']
    # jobs = sorted(jobs)
    # jobs_map = dict(zip(jobs, list(range(len(jobs)))))
    # return jobs_map[x]

    """第二种"""
    if x in ['technician', 'admin.', 'unknown', 'self-employed']:  # 差不多持平
        return 0
    elif x in ['blue-collar', 'services', 'entrepreneur', 'housemaid']:  # 负样本多余正样本
        return 1
    elif x in ['management', 'retired', 'student', 'unemployed']:  # 正样本多余负样本
        return 2


def education_transfer(x):
    """实现教育水平到数值的转换"""
    edu_level = ['primary', 'secondary', 'tertiary', 'unknown']
    edu_map = dict(zip(edu_level, list(range(len(edu_level)))))
    return edu_map[x]


def marital_transfer(x):
    """实现婚姻状况到数值的转换"""
    marital_condition = ['married', 'divorced', 'single']
    marital_map = dict(zip(marital_condition, list(range(len(marital_condition)))))
    return marital_map[x]


def binary_transfer(x):
    """实现二类的数值转换"""
    return 0 if x == 'no' else 1


def contact_transfer(x):
    if x == 'cellular':
        return 0
    elif x == 'telephone':
        return 1
    else:
        return 2


def month_transfer(x):
    """第一种"""
    orderList = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_map = dict(zip(orderList, list(range(len(orderList)))))

    return month_map[x]

    # """第二种"""
    # if x in ['jan', 'feb', 'mar', 'apr']:
    #     return 0
    # elif x in ['may', 'jun', 'jul', 'aug']:
    #     return 1
    # else:
    #     return 2

    # """第三种"""
    # if x in ['mar', 'sep', 'dec', 'oct']:
    #     return 0
    # elif x in ['apr', 'feb']:
    #     return 1
    # elif x in ['jan', 'aug', 'nov']:
    #     return 2
    # elif x in ['jun', 'jul']:
    #     return 3
    # else:
    #     return 4


def poutcome_transfer(x):
    if x == 'success':
        return 0
    elif x in 'failure':
        return 1
    elif x == 'other':
        return 2
    else:
        return 3


def age_transfer(x):
    if x <= 31.0 or x > 52.0:
        return 1
    elif x <= 39.0:
        return 2
    else:
        return 3


def balance_transfer(x):
    if x <= 175.0:
        return 0
    elif x <= 967.333:
        return 1
    else:
        return 2


def duration_transfer(x):
    if x <= 103:
        return 0
    elif x <= 181.0:
        return 1
    elif x <= 317.0:
        return 2
    else:
        return 3


def campaign_transfer(x):
    if x == 1:
        return 0
    else:
        return 1


def transfer_df(data_df):
    """data_df为DataFrame结构，这个函数用于将其中一些字段的值进行转换"""
    # 类别属性
    data_df['job'] = data_df['job'].apply(job_transfer)
    data_df['marital'] = data_df['marital'].apply(marital_transfer)
    data_df['education'] = data_df['education'].apply(education_transfer)
    data_df['loan'] = data_df['loan'].apply(binary_transfer)
    data_df['housing'] = data_df['housing'].apply(binary_transfer)
    data_df['isLoan'] = data_df['loan'] + data_df['housing']
    data_df['default'] = data_df['default'].apply(binary_transfer)
    data_df['contact'] = data_df['contact'].apply(contact_transfer)
    data_df['poutcome'] = data_df['poutcome'].apply(poutcome_transfer)
    data_df['month'] = data_df['month'].apply(month_transfer)

    # 数值属性
    data_df['age'] = data_df['age'].apply(age_transfer)
    data_df['balance'] = data_df['balance'].apply(balance_transfer)
    data_df['duration'] = data_df['duration'].apply(duration_transfer)
    data_df['campaign'] = data_df['campaign'].apply(campaign_transfer)
    data_df['pdays'] = data_df['pdays'].apply(lambda x: 0 if x == -1 else 1)
    data_df['previous_minus_campaign'] = abs(data_df['previous'] - data_df['campaign'])
    data_df['previous'] = data_df['previous_minus_campaign'].apply(lambda x: 0 if x <= 1 else 1)

    # 删除多余的列
    # data_df.drop(columns=['previous_minus_campaign', 'housing', 'loan'], inplace=True)

    return data_df
