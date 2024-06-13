# 各国や各組織の進路予報を使用することを想定して一応残しておく
# 予想したデータを受け渡す役割を持つのでForecasterとした
# 現状は真のデータから期間分のデータを取り出すこと以外の能力はない

import random

import polars as pl
from geopy.distance import geodesic


class Forecaster:
    """
    ############################## class Forecaster ##############################

    [ 説明 ]

    このクラスは台風発電船が行動を判断するのに用いる「予報データ」を作成するものです。

    クラス変数で予報の範囲と大まかな精度を指定することができます。

    ##############################################################################

    引数 :
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)

    属性 :
        forecast_time (int) : 予報の範囲(期間)を設定する。単位は時間[hours]。
        slope (float) : 予報の誤差距離を設定する。時間を変数とする1次関数の傾きであり遠い時間
                        ほど誤差が大きくなる。
        original_data (dataflame) : 過去の台風の座標とそれに対応する時刻を保有するデータ

    #############################################################################
    """

    def __init__(self, forecast_time, forecast_error_slope) -> None:
        self.forecast_time = forecast_time
        self.slope = forecast_error_slope

    def cal_error_radius_km(self, time_step, advance_time_hour):
        """
        ########################## def cal_error_radius_km ##########################

        [ 説明 ]

        誤差距離[km]を計算する関数です。前提として、時間の進み幅分1つ進んだ(例：時間の進み幅が6時間

        なら6時間後)の時間の誤差距離は0kmとします。設定した傾き(slope)の1次関数で現在時刻から進めた

        時間より誤差距離を算出します。

        予報の座標を出す際に正規分布に従ってoriginal_dataの座標からランダムに点を出しますが、その正

        規分布の設定における標準偏差の算出に使われる数値となります。

        ##############################################################################

        引数 :
            advance_time_hour (int) : シミュレーション上の現在時刻に対し進めた時間[hours]

        戻り値 :
            error_radius_km (int) : その時刻の正確な座標からの平均的な座標のずれ[km]

        #############################################################################
        """

        error_radius_km = self.slope * (advance_time_hour - time_step)
        return error_radius_km

    # sd : standerd deviation 標準偏差
    def cal_forecast_point_lat_sd(self, error_radius_km, original_point):
        """
        ####################### def cal_forecast_point_lat_sd #######################

        [ 説明 ]

        関数 create_forecast で用いる緯度の標準偏差を計算するための関数です。

        誤差距離がkm単位で与えられるのに対し、正規分布で座標を求めるために緯度経度の度数が単位の数値

        が必要となります。ここでは誤差距離をkm単位から度数単位に変換することを行なっています。これが

        求める標準偏差そのものになります。適度な度数の幅を定め、その幅で緯度のみをずらしながら誤差距

        離に近くなるまでずらします。誤差距離に近くなった時の緯度とずらされる前の緯度の差がkmを度数に

        置き換えた数値になります。これを緯度の標準偏差として用います。

        わざわざこのようにして求めているのは緯度1度、経度1度の長さは緯度によって変わってしまうからで

        す。緯度は比較的差が小さいですが経度は大きいため緯度に対応し都度求める必要があります。

        ##############################################################################

        引数 :
            error_radius_km (int) : その時刻の正確な座標からの平均的な座標のずれ[km](誤差距離)
            original_point (tuple) : original_dataにおける該当時刻の台風の座標


        戻り値 :
            lat_sd (float) : 緯度の標準偏差として用いられる値。distanceが誤差距離にほぼ等しい時
                             のtemp_latとoriginal_point_latの差

        #############################################################################
        """

        original_point_lat = original_point[0]
        original_point_lon = original_point[1]
        distance = 0

        # 緯度1度分の距離がだいたい110〜112km
        # 1kmの場合の度数を(1/112)とし、誤差距離の30分の1ずつ調べることとする
        split_deg_num = (1 / 112) * (error_radius_km / 30)
        temp_lat = original_point_lat

        while distance < error_radius_km:
            temp_lat = temp_lat + split_deg_num
            temp_point = (temp_lat, original_point_lon)

            distance = geodesic(original_point, temp_point).km

        lat_sd = temp_lat - original_point_lat

        return lat_sd

    def cal_forecast_point_lon_sd(self, error_radius_km, original_point):
        """
        ####################### def cal_forecast_point_lon_sd #######################

        [ 説明 ]

        関数 create_forecast で用いる経度の標準偏差を計算するための関数です。

        誤差距離がkm単位で与えられるのに対し、正規分布で座標を求めるために緯度経度の度数が単位の数値

        が必要となります。ここでは誤差距離をkm単位から度数単位に変換することを行なっています。これが

        求める標準偏差そのものになります。適度な度数の幅を定め、その幅で経度のみをずらしながら誤差距

        離に近くなるまでずらします。誤差距離に近くなった時の経度とずらされる前の経度の差がkmを度数に

        置き換えた数値になります。これを経度の標準偏差として用います。

        わざわざこのようにして求めているのは緯度1度、経度1度の長さは緯度によって変わってしまうからで

        す。緯度は比較的差が小さいですが経度は大きいため緯度に対応し都度求める必要があります。

        #############################################################################

        引数 :
            error_radius_km (int) : その時刻の正確な座標からの平均的な座標のずれ[km](誤差距離)
            original_point (tuple) : original_dataにおける該当時刻の台風の座標


        戻り値 :
            lon_sd (float) : 緯度の標準偏差として用いられる値。distanceが誤差距離にほぼ等しい時
                             のtemp_lonとoriginal_point_lonの差

        #############################################################################
        """

        original_point_lat = original_point[0]
        original_point_lon = original_point[1]
        distance = 0

        # 経度1度分の距離がだいたい北緯0で112km、北緯70で38km
        # 1kmの場合の度数を(1/112)とし、誤差距離の30分の1ずつ調べることとする
        split_deg_num = (1 / 112) * (error_radius_km / 30)
        lon = original_point_lon

        while distance < error_radius_km:
            lon = lon + split_deg_num
            temp_point = (original_point_lat, lon)

            distance = geodesic(original_point, temp_point).km

        lon_sd = lon - original_point_lon

        return lon_sd

    def create_forecast(self, time_step, current_time):
        """
        ############################ def create_forecast ############################

        [ 説明 ]

        関数 cal_error_radius_km , cal_forecast_point_lat_sd , cal_forecast_point_lon_sd

        を用いて、「予報データ」を作成する関数です。各時刻の台風の座標における緯度と

        経度それぞれで、original_dataの座標を平均値と関数を用いて算出した標準偏差の正規分布からラ

        ンダムに数値を出します。これによって得られた座標を予想座標として時刻、台風の番号とセットで記

        録します。current_timeからforecast_time期間分に存在するデータに対し、この処理を行なった

        ものを「予報データ」とします。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]
            current_time (int) : シミュレーション上の現在時刻(unixtime)


        戻り値 :
            forecast_data (dataflame) : 時刻、台風の番号、予想座標を持つ予報データ

        #############################################################################
        """

        unix_forecast_time = self.forecast_time * 3600
        start_forecast_time = current_time + time_step * 3600
        last_forecast_time = current_time + unix_forecast_time

        forecast_true_data = self.original_data.filter(
            (pl.col("unixtime") >= start_forecast_time)
            & (pl.col("unixtime") <= last_forecast_time)
        )

        unix_list = []
        ty_num_list = []
        lat_list = []
        lon_list = []

        forecast_lat_list = []
        forecast_lon_list = []

        # unix,ty_num,lat,lonが少なくともあれば良い
        rep_num = len(forecast_true_data)

        if rep_num != 0:
            for i in range(rep_num):
                unix_list.append(forecast_true_data[i, "unixtime"])
                ty_num_list.append(forecast_true_data[i, "TYPHOON NUMBER"])
                lat_list.append(forecast_true_data[i, "LAT"])
                lon_list.append(forecast_true_data[i, "LON"])

                true_point = (lat_list[i], lon_list[i])

                advance_time_hour = (unix_list[i] - current_time) / 3600
                error_radius_km = self.cal_error_radius_km(time_step, advance_time_hour)

                lat_sd = self.cal_forecast_point_lat_sd(error_radius_km, true_point)
                lon_sd = self.cal_forecast_point_lon_sd(error_radius_km, true_point)

                forecast_lat = random.gauss(lat_list[i], lat_sd)
                forecast_lon = random.gauss(lon_list[i], lon_sd)

                forecast_lat_list.append(forecast_lat)
                forecast_lon_list.append(forecast_lon)

        forecast_data = pl.DataFrame(
            {
                "unixtime": unix_list,
                "TYPHOON NUMBER": ty_num_list,
                "TRUE_LAT": lat_list,
                "TRUE_LON": lon_list,
                "FORE_LAT": forecast_lat_list,
                "FORE_LON": forecast_lon_list,
            }
        )
        # forecast_data.columns=["unixtime","TYPHOON NUMBER","TRUE_LAT","TRUE_LON","FORE_LAT","FORE_LON"]

        return forecast_data
