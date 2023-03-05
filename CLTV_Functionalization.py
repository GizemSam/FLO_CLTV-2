import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from pandas import DataFrame
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

# 1. flo_data_20K.csv verisiniokuyunuz..Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)  # çeğrek değerleri hesapla
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range  # birici çeğerğin 1.5 üstü
    low_limit = quartile1 - 1.5 * interquantile_range  # 2. çeğerin 1.5 altı
    return low_limit, up_limit  # alt limit ve üst limiti göster


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

def create_cltv_p(dataframe, month):
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["total_num"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_val"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = pd.DataFrame({col: (pd.to_datetime(dataframe[col], format='%Y-%m-%d') if 'date' in col else dataframe[col]) for col in dataframe.columns})
    analysis_date = dataframe["last_order_date"].max() + datetime.timedelta(days=2)

    cltv_df: DataFrame = pd.DataFrame()

    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["total_num"]
    cltv_df["monetary_cltv_avg"] = dataframe["total_val"] / dataframe["total_num"]
    cltv_df.head()
    cltv_df.columns = ["customer_id", 'recency', 'T', 'frequency', 'monetary']

    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary'])
    cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'],
                                                  cltv_df['monetary'],
                                                  time=month,  # 6 aylık
                                                  freq="W",  # T'nin frekans bilgisi.
                                                  discount_rate=0.01)

    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_df

create_cltv_p(df,month=6).sort_values(by="cltv",ascending= False)