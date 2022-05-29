import pandas as pd
import shapely.wkt
import numpy as np
from config import my_config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class App:
    def __init__(self):
        self.path = my_config['PATH']
        self.df_roads = pd.read_csv(self.path + "/roads_dataset.csv")
        self.df_transport = pd.read_csv(self.path + "routes_dataset.csv")
        self.df_population = pd.read_csv(self.path + "rosstat_population_all_cities.csv")
        self.df_isochrones = pd.read_csv(self.path + "isochrones_walk_dataset.csv")
        self.df_stops = pd.read_csv(self.path + "osm_stops.csv")
        self.df_companies = pd.read_csv(self.path + "osm_amenity.csv")
        self.target_data = pd.read_csv(self.path + 'target_hakaton_spb.csv', sep=';', encoding='windows-1251')
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder
        self.pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ])

    def preprocess(self):
        self.df_roads = self.df_roads.drop_duplicates('osmid').fillna('')
        self.df_transport = self.df_transport.drop_duplicates('name')
        self.df_population = self.df_population.drop_duplicates('geo_h3_10')
        self.df_isochrones = self.df_isochrones.drop_duplicates('geo_h3_10')
        self.df_stops = self.df_stops.drop_duplicates('osm_id').fillna('')
        self.df_companies = self.df_companies.drop_duplicates('geo_h3_10').fillna(0)

    def calculate_area(self):
        walking_ploygones = self.df_isochrones['walk_15min']
        polygones = walking_ploygones.apply(convert_to_poly)
        self.df_isochrones['area'] = polygones.apply(get_area)
        return self.df_isochrones

    def sort_by_area(self):
        df = self.df_isochrones[['geo_h3_10', 'city', 'area']]
        cities = ['Нижний Новгород', 'Новосибирск', 'Екатеринбург', 'Санкт-Петербург']
        df_array = []
        for elem in cities:
            df_city_by_area = df[df['city'] == elem].copy(deep=True)
            df_city_by_area['score'] = pd.cut(df_city_by_area['area'], 10, labels=range(1, 11))
            df_array.append(df_city_by_area.sort_values(by=['score', 'area'], ascending=False))
        df_city_by_area_res = pd.concat(df_array)
        return df_city_by_area_res

    def process_all(self):
        self.preprocess()
        self.df_isochrones = self.calculate_area()
        df_city_by_area_res = self.sort_by_area()
        df_stops_res = self.df_stops.groupby('geo_h3_10').count()
        df_stops_res.rename(columns={'city': 'stops_count'}, inplace=True)
        df_companies_stops = self.df_companies.merge(df_stops_res, on='geo_h3_10', how='left').fillna(0)
        df_prefinal = df_city_by_area_res.merge(df_companies_stops, on='geo_h3_10')
        df_final = df_prefinal.merge(self.df_population, how='left', on='geo_h3_10')
        df_with_target = df_final.merge(self.target_data, how='outer').set_index('geo_h3_10')
        cols = ['area', 'score',
                'Автозапчасти для иномарок', 'Авторемонт и техобслуживание (СТО)',
                'Алкогольные напитки', 'Аптеки', 'Банки', 'Быстрое питание',
                'Доставка готовых блюд', 'Женская одежда', 'Кафе',
                'Косметика / Парфюмерия', 'Ногтевые студии', 'Овощи / Фрукты',
                'Парикмахерские', 'Платёжные терминалы', 'Постаматы',
                'Продуктовые магазины', 'Пункты выдачи интернет-заказов', 'Рестораны',
                'Страхование', 'Супермаркеты', 'Цветы', 'Шиномонтаж',
                'stops_count', 'population', 'atm_category', 'atm_cnt']
        return df_with_target[cols]

    def train(self, df):
        df = df.sample(frac=1)
        tmp_df = df[(df['area'] < 1) & (df.atm_cnt != df.atm_cnt)]
        rand_index =  np.random.choice([0, 1], size=(len(tmp_df['area']),), p=[31./32, 1./32])
        df_few = pd.concat([tmp_df[[bool(x) for x in rand_index]],
                            df[df.atm_cnt == df.atm_cnt]], axis=0)
        X_few = df_few.drop(['atm_category', 'atm_cnt'], axis=1)
        y_few = ((df_few.atm_cnt.fillna(0) != 0) + 0).astype('int')
        print(len(y_few))
        reg = LogisticRegression()
        scores = cross_val_score(reg,
                                 self.pipe.fit_transform(X_few), y_few.values, cv=10, scoring='accuracy')
        print(f"cross val score = {np.mean(scores)}")
        reg.fit(self.pipe.fit_transform(X_few), y_few.values)

        y_cat = df_few['atm_category']
        X_cat = X_few[y_cat == y_cat]
        y_cat = y_cat[y_cat == y_cat]
        reg_cat = LogisticRegression()
        reg_cat.fit(self.pipe.fit_transform(X_cat), self.encoder().fit_transform(y=y_cat))

        return reg, reg_cat

    def test(self, df, reg, reg_cat):
        X, y = df.drop(['atm_category', 'atm_cnt'], axis=1), df['atm_cnt'].fillna(0)
        X = self.pipe.fit_transform(X)
        is_high_prob = np.sum(reg.predict_proba(X) > 0.99, axis=1)
        df_high_prob = df[[bool(x) for x in is_high_prob]]
        X_high, y_high = df_high_prob.drop(['atm_category', 'atm_cnt'], axis=1), df_high_prob['atm_cnt'].fillna(0)
        X_high = self.pipe.fit_transform(X_high)
        df_cat = reg_cat.predict(X_high)
        df_reg = reg.predict(X_high)
        df_final = pd.concat([pd.Series(df_cat, name='atm_cat'),
                              pd.Series(df_reg, name='atm_predict'), pd.Series(y_high.values, name='atm_estimate')],
                             axis=1)
        df_final = pd.concat([df_final, df_high_prob.reset_index()], axis=1)
        df_final = df_final[df_final.atm_predict == df_final.atm_estimate + 1]
        df_final = df_final[(df_final.population > 500) & (df_final['Банки'] == 0)][
            ['geo_h3_10', 'atm_predict', 'atm_cat']]
        print(len(df_final))
        df_final.to_csv("final_table.csv", encoding='utf-8', index=False, sep=',')


def convert_to_poly(string):
    return shapely.wkt.loads(string)


def get_area(polygon):
    return polygon.area
