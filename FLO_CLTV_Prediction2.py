##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from pandas import DataFrame

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

# 1. flo_data_20K.csv verisiniokuyunuz..Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

df.isnull().sum()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.

df.describe().T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99) # çeğrek değerleri hesapla
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range # birici çeğerğin 1.5 üstü
    low_limit = quartile1 - 1.5 * interquantile_range #2. çeğerin 1.5 altı
    return low_limit, up_limit # alt limit ve üst limiti göster

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] =  round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin

#aykırı değerleri varsa baskılayanız.
"""replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")"""

columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
           "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df,col)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_val"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]



# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes

"""date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime"""

df = pd.DataFrame({col: (pd.to_datetime(df[col], format='%Y-%m-%d') if 'date' in col else df[col]) for col in df.columns})

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
import datetime
analysis_date = df["last_order_date"].max() + datetime.timedelta(days=2)    #2gün sonrası

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

cltv_df: DataFrame = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"]=((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"]=((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["total_num"]
cltv_df["monetary_cltv_avg"] = df["total_val"]/df["total_num"]
cltv_df.head()
cltv_df.columns = ["customer_id",'recency', 'T', 'frequency', 'monetary']

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


# Gerçek değer ile modeli karşılaştırma
from lifetimes.plotting import plot_period_transactions
import matplotlib.pyplot as plt

plot_period_transactions(bgf)

plt.show(block=True)

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

cltv_df.sort_values(by = "exp_sales_6_month", ascending=False).head(10)




# 2.  Gamma-Gamma modelini fit ediniz.
# Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])


# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv_df.head()

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values(by = "cltv", ascending=False).head(20)

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################


# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"]) #qcut fonkisyonu küçükten buyuge sıralar.


# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

cltv_df.groupby("cltv_segment").agg({"recency":"mean",
                                     "monetary":"mean",
                                     "frequency":"mean"})


#4 grupiçerisindenseçeceğiniz2 grupiçinyönetimekısakısa6 aylıkaksiyonönerilerindebulununuz

"""Verilen verilere göre, A ve D segmentleri arasında bir fark var. A segmenti daha yüksek ortalama satın alma miktarı, 
daha yüksek frekans ve daha düşük recency değerlerine sahiptir. 
D segmenti ise daha düşük ortalama satın alma miktarına, daha düşük frekansa ve daha yüksek recency değerine sahiptir.

A segmentindeki müşteriler, daha sık ve daha yüksek miktarlarda satın alma yaparlar.
Bu nedenle, A segmentindeki müşteriler için, daha yüksek fiyatlı ürünlere veya
müşterilerin daha sık satın alma yapabilecekleri daha sıkıştırılmış bir satın alma programına
(örneğin, abonelik hizmeti) yönelik pazarlama kampanyaları düzenlenebilir.

D segmentindeki müşteriler ise daha düşük miktarlarda ve daha az sıklıkta satın alma yaparlar.
Bu müşterilerin daha fazla teşvik edilmesi ve daha sık satın alma yapmaları sağlanmalıdır.
Bu nedenle, D segmentindeki müşteriler için, özel indirimler, promosyonlar veya ödüllendirme programları
gibi teşvikler sağlayan pazarlama kampanyaları düzenlenebilir. 
Ayrıca, müşterilerin sadakatlerinin artırılması için daha fazla kişiselleştirilmiş hizmetler ve destekler sağlanabilir."""

