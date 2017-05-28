# coding: utf-8
__author__ = "nky510"
"""
ここにファイルの定義を記述
"""
import pandas as pd
import requests
import csv


def fetch_nikkei(start_drop="2017-04-01", end_drop="2017-05-27", use_past=False):
    """
    
    :param str start_drop: 
    :param str end_drop: 
    :param bool use_past: 
    :return:
    """
    header, data = download_nikkei(2014, 2017)
    df = pd.DataFrame(data, columns=header, dtype=float)
    df["日付"] = pd.to_datetime(df["日付"])
    df = df.sort_values("日付")
    df = df.reset_index(drop=True)
    df = df.rename(columns={"終値": "target"})
    use_cols = []
    df["days"] = _make_days_feature(df["日付"])
    use_cols.append("days")

    # 過去 10 日分の終値を特徴量としてつかう
    if use_past:
        for i in range(1, 11):
            n = "target_b{i}".format(**locals())
            df[n] = None
            df.loc[i:, n] = df.target.iloc[:-i].values
            use_cols.append(n)

    df = df.dropna()
    idx_train = (df["日付"] < pd.to_datetime(start_drop)) | (df["日付"] > pd.to_datetime(end_drop))

    X = df[use_cols]
    y = df[["target"]]
    x_train, y_train = X[idx_train].values, y[idx_train].values
    x_test = X.values
    return x_train, y_train, x_test


def _make_days_feature(datetime):
    """
    
    :param pd.Series datetime: 
    :return:
    :rtype: pd.Series
    """
    start = datetime.min()
    return (datetime - start).astype("int") / 86400000000000


def download_nikkei(start, end):
    """
    日経平均のダウンロード
    
    :param start: 
    :param end: 
    :return: 
    :rtype: tuple of (list<str>, list)
    """
    data = None
    header = None
    base_url = "http://k-db.com/indices/I101/1d/"
    urls = [base_url + str(i) for i in range(start, end + 1)]
    for url in urls:
        with requests.Session() as s:
            r = s.get(url, params={"download": "csv"})
            decoded = r.content.decode("shift-jis")

        cr = csv.reader(decoded.splitlines(), delimiter=",")
        cr = list(cr)
        if header is None:
            header = cr[0]

        if data is None:
            data = cr[1:]
        else:
            data.extend(cr[1:])
    return header, data
