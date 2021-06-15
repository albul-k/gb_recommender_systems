from typing import Iterable, Optional
import pandas as pd
import numpy as np
import random

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data: pd.DataFrame, weighting_type: Optional[str] = None) -> None:
        """Initialization

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        weighting_type : str, optional
            Possible values is `bm25_weight` or `tfidf_weight`
        """
        
        self.user_item_matrix = self.prepare_matrix(data) 
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts()
        
        if weighting_type == 'bm25_weight':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        elif weighting_type == 'tfidf_weight':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit()
        self.own_recommender = self.fit_own_recommender()
     
    def prepare_matrix(self, data: pd.DataFrame) -> pd.pivot_table:
        user_item_matrix = pd.pivot_table(
            data, 
            index='user_id',
            columns='item_id', 
            values='quantity',
            aggfunc='count', 
            fill_value=0
        )
        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        return user_item_matrix
    
    def prepare_dicts(self) -> Iterable[dict]:
        """Подготавливает вспомогательные словари"""
        
        userids = self.user_item_matrix.index.values
        itemids = self.user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    def fit_own_recommender(self) -> ItemItemRecommender():
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return own_recommender
    
    def fit(self, n_factors: int = 20, regularization: float = 0.001, iterations: int = 15, num_threads: int = 4) -> AlternatingLeastSquares():
        """Train ALS

        Parameters
        ----------
        n_factors : int, optional
            [description], by default 20
        regularization : float, optional
            [description], by default 0.001
        iterations : int, optional
            [description], by default 15
        num_threads : int, optional
            [description], by default 4

        Returns
        -------
        [type]
            [description]
        """
        
        model = AlternatingLeastSquares(
            factors=n_factors,
            regularization=regularization,
            iterations=iterations,
            num_threads=num_threads
        )
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N: int = 5) -> list:
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        recs: list = [self.id_to_itemid[rec[0]] for rec in 
            self.own_recommender.recommend(
                userid=self.userid_to_id[user], 
                user_items=csr_matrix(self.user_item_matrix).tocsr(),
                N=N, 
                filter_already_liked_items=False, 
                filter_items=None, 
                recalculate_user=True
            )
        ]
        print(recs)
        res: list = list()
        for item_row_id in recs:
            row_id, _ = self.model.similar_items(self.id_to_itemid[item_row_id], N=1)
            res.append(self.id_to_itemid[row_id])
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user: int, N: int = 5) -> list:
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # берем N похожих юзеров
        users: list = [self.userid_to_id[rec[0]] for rec in 
            self.model.similar_users(self.userid_to_id[user], N=N)
        ]

        # от каждого похожего юзера берем N товаров
        res: list = list()
        for user_ in users:
            recs: list = [self.id_to_itemid[rec[0]] for rec in 
                self.own_recommender.recommend(
                    userid=self.userid_to_id[user_], 
                    user_items=csr_matrix(self.user_item_matrix).tocsr(),
                    N=N, 
                    filter_already_liked_items=False, 
                    filter_items=None, 
                    recalculate_user=True
                )
            ]

            for itm in recs:
                res.append(itm)
        
        # слйчайно выбираем N товаров
        res = random.choices(list(set(res)), k=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res