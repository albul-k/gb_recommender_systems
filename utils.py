"""
Utils

"""

import pandas as pd
import numpy as np


def prefilter_items(data: pd.DataFrame, item_features: pd.DataFrame, take_n_popular: int = 5000, interesting_cats: list = None) -> pd.DataFrame:
    """Prefilter function

    Parameters
    ----------
    data : pd.DataFrame
        Input user data
    item_features : pd.DataFrame
        Input item data
    take_n_popular : int, optional
        [description], by default 5000
    interesting_cats : list, optional
        [description], by default None

    Returns
    -------
    pd.DataFrame
        Filtered data
    """

    data_ = data.copy()

    # Уберем самые популярные товары (их и так купят)
    popularity = data_.groupby('item_id')['user_id'].nunique().reset_index() / data_['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data_ = data_[~data_['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data_ = data_[~data_['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    data_ = data_[~(data_['day'] > 365)]

    # Уберем не интересные для рекоммендаций категории (department)
    if len(interesting_cats) > 0:
        items_of_interest = item_features[item_features['department'].isin(interesting_cats)]
        data_ = pd.merge(data_, items_of_interest, on='item_id', how='inner')

    data_['price'] = data_['sales_value'] / (np.maximum(data_['quantity'], 1))

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    data_ = data_[~(data_['price'] < data_['price'].quantile(0.20))]

    # Уберем слишком дорогие товары
    data_ = data_[~(data_['price'] > data_['price'].quantile(0.99995))]

    return data_
    
def postfilter_items(user_id, recommednations):
    pass