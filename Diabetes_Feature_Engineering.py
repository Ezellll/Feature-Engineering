import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

###########################################################
# Görev 1 : EDA
###########################################################

# Adım 1: 

def load():
    data = pd.read_csv("feature_engineering/diabetes/diabetes.csv")
    return data

df = load()
df.columns
df.head()
df.info()
df.shape

# Diabetes Pedigree Function = Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon



# Adım 2: Numerik ve kategorik değişkenler gösterildi ve yakalandı.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # Nümerik görünülü kategorikleri çıkarttık
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizi yapıldı.

df[num_cols].head()
df[num_cols].describe().T
# kategorik ---> outcome
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)


# Adım 4: Hedef değişken analizi yapıldı. (Kategorik değişkenlere göre hedef
# değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

df.groupby("Outcome")[num_cols].mean()

df.groupby(cat_cols)["Outcome"].mean()

# Adım 5: Aykırı gözlem analizi yapıldı.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

df[num_cols].describe([.25, .5, .75, .90, .95, .99]).T

for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 6: Eksik gözlem analizi yapıldı.

df.isnull().values.any()
df.isnull().sum()
df.notnull().sum()
df.shape

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)


# Adım 7: Korelasyon analizi yapıldı.

df.corr()
df.corrwith(df["Outcome"]).sort_values(ascending=False)

###############################################################################
# Görev 2 :  Feature Engineering
###############################################################################

# Adım 1: Eksik ve aykırı değerler için doldurma, baskılama işlemleri yapıldı.
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır.
# Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak
# atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
na_Columns = missing_values_table(df, True)
df[na_Columns].describe().T

df["SkinThickness"].fillna(df.groupby("Pregnancies")["SkinThickness"].transform("median"), inplace=True)
df["BMI"].fillna(df["BMI"].mean(), inplace=True)

# Adım 2 de oluşturulan BMI_Class kırlımının ortalamaları ile doldurduk
df.loc[(df["BMI"] <= 18.4), "BMI_Class"] = "Zayıf"
df.loc[(df["BMI"] > 18.4) & (df["BMI"] <= 24.9), "BMI_Class"] = "Normal"
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "BMI_Class"] = "Fazla Kilolu"
df.loc[(df["BMI"] > 29.9), "BMI_Class"] = "Obez"
df["BloodPressure"].fillna(df.groupby("BMI_Class")["BloodPressure"].transform("median"), inplace=True)

df["Insulin"].fillna(df.groupby("Outcome")["Insulin"].transform("median"), inplace=True)
df["Glucose"].fillna(df.groupby("Outcome")["Glucose"].transform("median"), inplace=True)


"""
from sklearn.impute import KNNImputer
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
"""


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

df.head()
df[num_cols].describe([.25, .5, .75, .90, .95, .99]).T
# df["Pregnancies"].describe([.25, .5, .75, .90, .95, .99]).T

for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 2: Create new columns.

df.columns
df.loc[(df["BMI"] < 18.4), "BMI_Class"] = "Zayıf"
df.loc[(df["BMI"] > 18.5) & (df["BMI"] < 24.9), "BMI_Class"] = "Normal"
df.loc[(df["BMI"] > 24.9) & (df["BMI"] < 29.9), "BMI_Class"] = "Fazla Kilolu"
df.loc[(df["BMI"] > 30), "BMI_Class"] = "Obez"

df.loc[((df["Insulin"] < 16) | (df["Insulin"] > 166)), "Insulin_Range"] = "Anormal"
df.loc[(df["Insulin"] > 16) & (df["Insulin"] < 166), "Insulin_Range"] = "Normal"

df.loc[((df["Glucose"] < 70)), "Glucose_Range"] = "Low"
df.loc[(df["Glucose"] > 70) & (df["Glucose"] < 99), "Glucose_Range"] = "Normal"
df.loc[(df["Glucose"] > 99) & (df["Glucose"] < 126), "Glucose_Range"] = "Secret"
df.loc[(df["Glucose"] > 126) & (df["Glucose"] < 200), "Glucose_Range"] = "High"


df.groupby("BMI_Class")["Outcome"].mean()
df.groupby("Insulin_Range")["Outcome"].mean()
df.groupby("Glucose_Range")["Outcome"].mean()

df.head()

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# zayıf olanlarda normalleri rare edebilirim



def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
label_encoder(df, "Insulin_Range")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
# Burası farklı olcak
df = one_hot_encoder(df, ["BMI_Class" , "Glucose_Range", "Insulin_Range"])
df.head()

# Adım 4: Standardization

mms = MinMaxScaler()

df[num_cols] = mms.fit_transform(df[num_cols])

# Adım 5: Modelling.

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)



