import math

import numpy as np
import polars as pl
from geopy.distance import geodesic


class TPGship:
    """
    ############################### class TPGship ###############################

    [ 説明 ]

    このクラスは台風発電船を作成するクラスです。台風発電船の行動とその行動をするための条件を定義します。

    台風発電船自体の能力や状態量もここで定義されることになります。

    ##############################################################################

    引数 :
        year (int) : シミュレーションを行う年
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)

    属性 :
        max_storage (float) : 台風発電船の蓄電容量の上限値
        generator_output (float) : 台風発電船の定格出力
        wind_propulsion_power (float) : 通常海域で台風発電船の風力推進機で得られる平均出力
        generating_facilities_need_max_power (float) : 停止状態の発電機を船体と最大船速で推進させるために必要な出力
        max_speed_power (float) : 付加物のない船体を最大船速で進めるのに必要な出力


        storage (float) : 台風発電船のその時刻での蓄電量
        storage_percentage (float) : 台風発電船のその時刻での蓄電量の割合
        gene_elect (float) : その時刻での発電量
        loss_elect (float) : その時刻での消費電力量
        ship_state (int) : 台風発電船の状態。通常航行、待機=0,発電状態=1,台風追従=2,台風低速追従=2.5,拠点回航=3,待機位置回航=4。
        total_gene_elect (float) : その時刻までの合計発電量
        total_loss_elect (float) : その時刻までの合計消費電力量
        total_gene_time (int) : その時刻までの合計発電時間
        total_loss_time (int) : その時刻までの合計電力消費時間

        speed_kt (float) : その時刻での台風発電船の船速(kt)
        target_name (str) : 目標地点の名前。台風の場合は番号の文字列入力。
        base_lat (float) : 拠点の緯度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        base_lon (float) : 拠点の経度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        standby_lat (float) : 待機位置の緯度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        standby_lon (float) : 待機位置の経度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        ship_lat (float) : その時刻での台風発電船の緯度
        ship_lon (float) : その時刻での台風発電船の経度
        target_lat (float) : 目標地点の緯度
        target_lon (float) : 目標地点の経度
        target_distance (float) : 台風発電船から目標地点までの距離(km)
        target_TY (int) : 追従対象の台風の番号の整数入力。ない場合は0が入る。
        go_base (int) : 1の時その時の蓄電容量によらず拠点に帰る。
        next_TY_lat (float) : time_step後の目標台風の緯度。ない場合は経度と共に0
        next_TY_lon (float) : time_step後の目標台風の経度。ない場合は緯度と共に0
        next_ship_TY_dis (float) : time_step後の目標台風と台風発電船の距離(km)。ない場合はNaN。
        brance_condition (str) : 台風発電船が行動分岐のどの分岐になったかを示す

        distance_judge_hours (int) : 追従判断基準時間。発電船にとって台風が遠いか近いかを判断する基準。　※本プログラムでは使用しない
        judge_energy_storage_per (int) : 発電船が帰港判断をする蓄電割合。
        effective_range (float) : 発電船が台風下での航行となる台風中心からの距離[km]
        sub_judge_energy_storage_per (int) : 発電船が拠点経由で目的地に向かう判断をする蓄電割合。
        judge_direction (float) : 発電船が2つの目的地の方位差から行動を判断する時の基準値[度]
        standby_via_base (int) : 待機位置へ拠点を経由して向かう場合のフラグ
        judge_time_times (float) : 台風の補足地点に発電船が最大船速で到着する時間に対し台風が到着する時間が「何倍」である時追うと判断するのかの基準値

        normal_ave_speed (float) : 平常時の平均船速(kt)
        max_speed (float) : 最大船速(kt)
        TY_tracking_speed (float) : 台風を追いかける時の船速(kt)
        speed_kt (float) : その時の船速(kt)

        forecast_data (dataflame) : 各時刻の台風の予想座標がわかるデータ。台風番号、時刻、座標を持つ　※Forecasterからもらう必要がある。
        TY_start_time_list (list) : 全ての台風の発生時刻のリスト　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        forecast_weight (float) : 台風を評価する際の式で各項につける重みの数値。他の項は(100-forecast_weight)。　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。



    #############################################################################
    """

    ####################################  パラメータ  ######################################

    wind_propulsion_power = 0
    generating_facilities_need_max_power = 0
    max_speed_power = 0

    def __init__(
        self,
        first_locate,
        hull_num,
        storage_method,
        max_storage_wh,
        generator_output_w,
        ship_return_speed_kt,
        ship_max_speed_kt,
        forecast_weight,
        typhoon_effective_range,
        judge_energy_storage_per,
        sub_judge_energy_storage_per,
        judge_time_times,
    ) -> None:
        self.ship_lat = first_locate[0]
        self.ship_lon = first_locate[1]
        self.hull_num = hull_num
        self.storage_method = storage_method
        self.max_storage = max_storage_wh
        self.generator_output = generator_output_w
        self.nomal_ave_speed = ship_return_speed_kt
        self.max_speed = ship_max_speed_kt

        self.forecast_weight = forecast_weight
        self.effective_range = typhoon_effective_range
        self.judge_energy_storage_per = judge_energy_storage_per
        self.sub_judge_energy_storage_per = sub_judge_energy_storage_per
        self.judge_time_times = judge_time_times

    ####################################  状態量  ######################################

    # 状態量の初期値入力
    def set_initial_states(self):
        """
        ############################ def set_initial_states ############################

        [ 説明 ]

        台風発電船の各種状態量に初期値を与える関数です。

        max_storage , base_lat , base_lon , standby_lat , standby_lonの数値の定義が少なくとも先に必要です。

        ##############################################################################

        """

        # 船内電気関係の状態量
        self.storage = self.max_storage * 0.1
        self.storage_percentage = (self.storage / self.max_storage) * 100
        self.supply_elect = 0
        self.gene_elect = 0
        self.loss_elect = 0
        self.ship_state = 0
        self.total_gene_elect = 0
        self.total_loss_elect = 0
        self.total_gene_time = 0
        self.total_loss_time = 0

        # 発電船の行動に関する状態量(現状のクラス定義では外部入力不可(更新が内部関数のため))
        self.speed_kt = 0
        self.target_name = "base station"
        self.target_lat = self.base_lat
        self.target_lon = self.base_lon
        self.target_distance = 0
        self.target_TY = 0
        self.go_base = 0
        self.TY_and_base_action = 0
        self.next_TY_lat = 0
        self.next_TY_lon = 0
        self.next_ship_TY_dis = np.nan
        self.brance_condition = "start forecast"

        # 発電船自律判断システム設定
        self.judge_direction = 10
        self.standby_via_base = 0

    ####################################  メソッド  ######################################

    # 船の機能としては発電量計算、消費電力量計算、予報データから台風の目標地点の決定、timestep後の時刻における追従対象台風の座標取得のみ？
    # 状態量を更新するような関数はメソッドではない？

    # とりあえず状態量の計算をしている関数がわかるように　#状態量計算　をつけておく

    def calculate_power_consumption(self, time_step):
        """
        ############################ def calculate_power_consumption ############################

        [ 説明 ]

        time_stepごとの台風発電船の消費電力量(Wh)を計算する関数です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        戻り値 :
            energy_loss (float) : time_stepで消費される電力量[Wh]

        #############################################################################
        """

        # 台風追従に必要な出力
        typhoon_tracking_power = (
            self.max_speed_power + self.generating_facilities_need_max_power
        ) * ((self.speed_kt / self.max_speed) ** 3) - self.wind_propulsion_power

        if typhoon_tracking_power < 0:
            typhoon_tracking_power = 0

        # 電気から動力への変換は損失なしで行える仮定
        energy_loss = typhoon_tracking_power * time_step

        return energy_loss

    # 状態量計算
    def get_distance(self, target_position):
        """
        ############################ def get_distance ############################

        [ 説明 ]

        台風発電船から目標地点への距離を計算する関数です。

        ##############################################################################

        引数 :
            target_position (taple) : 目標地点の座標(緯度,経度)


        戻り値 :
            distance (float) : 台風発電船から目標地点への距離(km)

        #############################################################################
        """

        A_position = (self.ship_lat, self.ship_lon)

        # AーB間距離
        distance = geodesic(A_position, target_position).km

        return distance

    def get_direction(self, target_position):
        """
        ############################ def get_distance ############################

        [ 説明 ]

        台風発電船から目標地点への方位を計算する関数です。

        反時計回り(左回り)を正として角度（度数法）を返します。

        ##############################################################################

        引数 :
            target_position (taple) : 目標地点の座標(緯度,経度)


        戻り値 :
            direction (float) : 台風発電船から目標地点への方位(度)

        #############################################################################
        """
        # 北を基準に角度を定義する
        x1 = 10 + self.ship_lat  # 北緯(10+船の緯度)度
        y1 = 0 + self.ship_lon  # 東経(0+船の経度)度

        # 外積計算　正なら左回り、負なら右回り
        # 船の座標 (回転中心)
        x2 = self.ship_lat
        y2 = self.ship_lon

        # 目標地点の座標
        x3 = target_position[0]
        y3 = target_position[1]

        gaiseki = (x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2)
        naiseki = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
        size12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        size32 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

        if gaiseki == 0:  # 直線上

            if naiseki < 0:
                direction = np.pi
            else:
                direction = 0

        elif gaiseki < 0:  # 右回り
            direction = -np.arccos((naiseki) / (size12 * size32))

        elif gaiseki > 0:  # 左回り
            direction = np.arccos((naiseki) / (size12 * size32))

        else:
            print("direction error")

        direction = np.rad2deg(direction)

        return direction

    # 状態量計算
    def change_kt_kmh(self):
        """
        ############################ def change_kt_kmh ############################

        [ 説明 ]

        ktをkm/hに変換する関数です

        ##############################################################################


        戻り値 :
            speed_kmh (float) : km/hに変換された船速

        #############################################################################
        """

        speed_kmh = self.speed_kt * 1.852

        return speed_kmh

    # 予報データから台風の目標地点の決定
    def get_target_data(self, year, current_time, time_step):
        """
        ############################ def get_target_data ############################

        [ 説明 ]

        「予報データ」(forecast_data)から目標とする台風を決める関数です。

        予想発電時間は台風発電船が台風に追いついてから台風が消滅するまで追った場合の時間です。

        消滅時間がわからない場合は発生から5日後に台風が消滅するものとして考えます。5日以上存在する場合は予報期間の最後の時刻に消滅すると仮定します。

        台風補足時間は台風発電船が予想される台風の座標に追いつくまでにかかる時間です。

        以上二つの数値を用いて評価用の数値を以下のように計算します。

        評価数値　＝　予想発電時間＊(forecast_weight) - 台風補足時間＊(100 - forecast_weight)

        これを予報データ内の全データで計算して最も評価数値が大きかったものを選びそれを返します。

        2023/05/24追記

        補足時間について、台風発電船の最大船速で到着するのにかかる時間のX倍の時間をかけなければ台風の想定到着時間に目的地に到着できない場合、

        選択肢から除外するものとする。

        Xは判断の基準値として設定されるものとする。

        ##############################################################################

        引数 :
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        戻り値 :
            target_typhoon_data (dataflame) : 追従目標に選ばれた台風のデータ。予報データから1行分のみ取り出した状態。

        #############################################################################
        """

        # 台風の平均存続時間
        # 今回は5日ととりあえずしてある
        TY_mean_time_to_live = 24 * 5
        TY_mean_time_to_live_unix = TY_mean_time_to_live * 3600

        ship_speed_kmh = self.change_kt_kmh()

        # unixtimeでの時間幅
        forecast_time_unix = 3600 * self.forecast_time
        last_forecast_time = int(current_time + forecast_time_unix)
        start_forecast_time = int(current_time + 3600 * time_step)

        # 該当時刻内のデータの抜き出し
        typhoon_data_forecast = self.forecast_data

        # 陸地認識フェーズ　陸地内に入っているデータの消去
        typhoon_data_forecast = typhoon_data_forecast.filter(
            (
                ((pl.col("FORE_LAT") >= 0) & (pl.col("FORE_LAT") <= 13))
                & (pl.col("FORE_LON") >= 127.5)
            )  # p1 ~ p2
            | (
                ((pl.col("FORE_LAT") >= 13) & (pl.col("FORE_LAT") <= 15))
                & (pl.col("FORE_LON") >= 125)
            )  # p25 ~ p255
            | (
                ((pl.col("FORE_LAT") >= 15) & (pl.col("FORE_LAT") <= 24))
                & (pl.col("FORE_LON") >= 123)
            )  # p3 ~ p4
            | (
                ((pl.col("FORE_LAT") >= 24) & (pl.col("FORE_LAT") <= 26))
                & (pl.col("FORE_LON") >= 126)
            )  # p5 ~ p55
            | (
                ((pl.col("FORE_LAT") >= 26) & (pl.col("FORE_LAT") <= 28))
                & (pl.col("FORE_LON") >= 130.1)
            )  # p555 ~ p6
            | (
                ((pl.col("FORE_LAT") >= 28) & (pl.col("FORE_LAT") <= 32.2))
                & (pl.col("FORE_LON") >= 132.4)
            )  # p7 ~ p8
            | (
                ((pl.col("FORE_LAT") >= 32.2) & (pl.col("FORE_LAT") <= 34))
                & (pl.col("FORE_LON") >= 137.2)
            )  # p9 ~ p10
            | (
                ((pl.col("FORE_LAT") >= 34) & (pl.col("FORE_LAT") <= 41.2))
                & (pl.col("FORE_LON") >= 143)
            )  # p11 ~ p12
            | (
                ((pl.col("FORE_LAT") >= 41.2) & (pl.col("FORE_LAT") <= 44))
                & (pl.col("FORE_LON") >= 149)
            )  # p13 ~ p14
            | (
                ((pl.col("FORE_LAT") >= 44) & (pl.col("FORE_LAT") <= 50))
                & (pl.col("FORE_LON") >= 156)
            )  # p15 ~ p16
            | ((pl.col("FORE_LAT") >= 50))  # p16 ~
        )

        # 台風番号順に並び替え
        typhoon_data_forecast = typhoon_data_forecast.select(
            pl.col("*").sort_by("TYPHOON NUMBER")
        )

        if len(typhoon_data_forecast) != 0:
            # 予報における一番若い番号の台風の取得
            TY_start_bangou = typhoon_data_forecast[0, "TYPHOON NUMBER"]
            TY_end_bangou = typhoon_data_forecast[
                len(typhoon_data_forecast) - 1, "TYPHOON NUMBER"
            ]

            # 台風発生時刻の取得番号
            occurrence_time_acquisition_num = TY_start_bangou - (year - 2000) * 100
            # 台風番号より台風の個数を調べる
            TY_num_forecast = typhoon_data_forecast.n_unique("TYPHOON NUMBER")

            # 予報期間内の台風がどの時刻まで予報されているのかを記録するリスト
            TY_forecast_end_time = []
            # 欠落した番号がいた場合のリスト
            missing_num_list = []
            # 各台風番号での予測終了時刻の取得
            TY_bangou = TY_start_bangou

            for i in range(TY_num_forecast):

                # 番号が後なのに先に発生しているケースがあったのでその応急処置
                # if (i == TY_num_forecast -1) and (error_num == 1):
                # print("ERROR",TY_bangou,TY_end_bangou,"unixtime",current_time,"~",last_forecast_time)
                typhoon_data_by_num = typhoon_data_forecast.filter(
                    pl.col("TYPHOON NUMBER") == TY_bangou
                )
                while len(typhoon_data_by_num) == 0:
                    if len(typhoon_data_by_num) == 0:
                        missing_num_list.append(TY_bangou)
                        TY_bangou = TY_bangou + 1
                        typhoon_data_by_num = typhoon_data_forecast.filter(
                            pl.col("TYPHOON NUMBER") == TY_bangou
                        )

                typhoon_data_by_num = typhoon_data_forecast.filter(
                    pl.col("TYPHOON NUMBER") == TY_bangou
                )
                typhoon_data_by_num = typhoon_data_by_num.select(
                    pl.col("*").sort_by("unixtime", descending=True)
                )
                TY_forecast_end_time.append(typhoon_data_by_num[0, "unixtime"])

                TY_bangou = TY_bangou + 1

            # 現在地から予測される台風の位置までの距離
            distance_list = []
            # 現在地から予測される台風の位置に到着する時刻
            ship_catch_time_list = []
            # 現在時刻から目的地に台風が到着するのにかかる時間
            arrival_time_list = []
            # 上記二つの時間の倍率
            time_times_list = []
            # 到着時から追従した場合に予測される発電量
            projected_elect_gene_time = []
            # 現在地から台風の位置に到着するのに実際必要な時刻
            true_ship_catch_time_list = []

            # 台風番号順に並び替えて当該時刻に発電船が到着した場合に最後まで追従できる発電時間を項目として作る
            # last_forecast_time(予報内の最終台風存続確認時刻)と最後の時刻が等しい場合には平均の存続時間で発電量を推定する
            # 今回は良い方法が思いつかなかったので全データから台風発生時刻を取得する。本来は発生時刻を記録しておきたい。

            # 台風発生時刻の取得
            # 台風発生時刻を入れておくリスト
            TY_occurrence_time = []
            # 各台風番号で開始時刻の取得
            TY_occurrence_time = self.TY_start_time_list

            data_num = len(typhoon_data_forecast)

            # nd_time_list = []
            # start_time_list = []
            # shori = []
            # データごとに予測発電時間を入力する
            for i in range(data_num):
                # 仮の発電開始時間
                gene_start_time = typhoon_data_forecast[i, "unixtime"]
                # 考える台風番号
                TY_predict_bangou = typhoon_data_forecast[i, "TYPHOON NUMBER"]

                adjustment_num = 0
                for j in range(len(missing_num_list)):
                    if TY_predict_bangou >= missing_num_list[j]:
                        adjustment_num = adjustment_num + 1

                # データ参照用の番号
                data_reference_num = (
                    TY_predict_bangou - TY_start_bangou - adjustment_num
                )

                # 当該台風の予報内での終了時刻
                end_time_forecast_TY = TY_forecast_end_time[data_reference_num]
                # 当該台風の発生時刻
                start_time_forecast_TY = TY_occurrence_time[
                    TY_predict_bangou - (year - 2000) * 100 - 1
                ]
                # start_time_list.append(start_time_forecast_TY)
                # 台風最終予想時刻による場合分け。予報期間終了時刻と同じ場合はその後も台風が続くものとして、平均存続時間を用いる。
                # 平均存続時間よりも長く続いている台風の場合は最終予想時刻までを発電するものと仮定する。
                if (end_time_forecast_TY == last_forecast_time) and (
                    (end_time_forecast_TY - start_time_forecast_TY)
                    < TY_mean_time_to_live_unix
                ):

                    # 予想される発電時間[hour]を出す
                    forecast_gene_time = (
                        start_time_forecast_TY
                        + TY_mean_time_to_live_unix
                        - gene_start_time
                    ) / 3600
                    # end_time_list.append(start_time_forecast_TY + TY_mean_time_to_live_unix)
                    # shori.append(1)

                else:

                    # 予想期間内で発電時間[hour]を出す
                    forecast_gene_time = (end_time_forecast_TY - gene_start_time) / 3600
                    # end_time_list.append(end_time_forecast_TY)
                    # shori.append(2)

                projected_elect_gene_time.append(forecast_gene_time)

            # データフレームに予想発電時間の項目を追加する
            # typhoon_data_forecast["処理"] = shori
            # typhoon_data_forecast["予想発電開始時間"] = start_time_list
            # typhoon_data_forecast["予想発電終了時間"] = end_time_list
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(projected_elect_gene_time).alias("FORE_GENE_TIME")
            )

            # 距離の判別させる
            for i in range(data_num):

                typhoon_posi_future = (
                    typhoon_data_forecast[i, "FORE_LAT"],
                    typhoon_data_forecast[i, "FORE_LON"],
                )
                ship_typhoon_dis = self.get_distance(typhoon_posi_future)

                # 座標間の距離から到着時刻を計算する
                if ship_speed_kmh == 0:
                    ship_speed_kmh = self.max_speed * 1.852
                ship_catch_time = math.ceil(ship_typhoon_dis / ship_speed_kmh)

                # 現時刻から台風がその地点に到達するまでにかかる時間を出す
                typhoon_arrival_time = int(
                    (typhoon_data_forecast[i, "unixtime"] - current_time) / 3600
                )

                # arrival_time_list.append(typhoon_arrival_time)

                # ship_catch_time_list.append(ship_catch_time)

                time_times_list.append(ship_catch_time / typhoon_arrival_time)

                # 台風の到着予定時刻と船の到着予定時刻の内遅い方をとる
                if typhoon_arrival_time > ship_catch_time:
                    true_ship_catch_time_list.append(typhoon_arrival_time)
                else:
                    true_ship_catch_time_list.append(ship_catch_time)

                distance_list.append(ship_typhoon_dis)

                # print(ship_typhoon_dis)
                # print(typhoon_data_forecast.loc[i,"distance"])

            # 台風の距離を一応書いておく
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(distance_list).alias("distance")
            )
            # データフレームに距離の項目を追加する
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(true_ship_catch_time_list).alias("TY_CATCH_TIME")
            )
            # データフレームに距離の項目を追加する
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(time_times_list).alias("JUDGE_TIME_TIMES")
            )

            # 予想発電時間と台風補足時間の差を出す
            time_difference = []
            for i in range(len(typhoon_data_forecast)):
                time_difference.append(
                    typhoon_data_forecast[i, "FORE_GENE_TIME"] * self.forecast_weight
                    - typhoon_data_forecast[i, "TY_CATCH_TIME"]
                    * (100 - self.forecast_weight)
                )

            # 予想発電時間と台風補足時間の差をデータに追加。名前は時間対効果
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(time_difference).alias("TIME_EFFECT")
            )

            # 基準倍数以下の時間で到達できる台風のみをのこす。[実際の到達時間] ≦ (台風の到着時間) が実際の判定基準
            typhoon_data_forecast = typhoon_data_forecast.filter(
                pl.col("JUDGE_TIME_TIMES") <= self.judge_time_times
            )

            # データを時間対効果が大きい順に並び替える
            typhoon_data_forecast = typhoon_data_forecast.select(
                pl.col("*").sort_by("TIME_EFFECT", descending=True)
            )

            if len(typhoon_data_forecast) != 0:
                # 出力データフレーム
                time_effect = typhoon_data_forecast[0, "TIME_EFFECT"]

                typhoon_data_forecast = typhoon_data_forecast.filter(
                    pl.col("TIME_EFFECT") == time_effect
                )

                if len(typhoon_data_forecast) > 1:
                    # データを発電時間が長い順に並び替える
                    typhoon_data_forecast = typhoon_data_forecast.select(
                        pl.col("*").sort_by("FORE_GENE_TIME", descending=True)
                    )

                    gene_time_max = typhoon_data_forecast[0, "FORE_GENE_TIME"]
                    typhoon_data_forecast = typhoon_data_forecast.filter(
                        pl.col("FORE_GENE_TIME") == gene_time_max
                    )

                    if len(typhoon_data_forecast) > 1:
                        # データを台風補足時間が短い順に並び替える
                        typhoon_data_forecast = typhoon_data_forecast.select(
                            pl.col("*").sort_by("TY_CATCH_TIME")
                        )

                        gene_time_max = typhoon_data_forecast[0, "TY_CATCH_TIME"]
                        typhoon_data_forecast = typhoon_data_forecast.filter(
                            pl.col("TY_CATCH_TIME") == gene_time_max
                        )

        return typhoon_data_forecast

    # timestep後の時刻における追従対象台風の座標取得
    def get_next_time_target_TY_data(self, time_step, current_time):
        """
        ############################ def get_next_time_target_TY_data ############################

        [ 説明 ]

        get_target_dataで選ばれ、追従対象となった台風のcurrent_time + time_stepの時刻での座標を取得する関数です。

        存在しない場合は空のデータフレームが返ります。

        ##############################################################################

        引数 :
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        戻り値 :
            next_time_target_data (dataflame) : 追従目標に選ばれた台風の次の時刻でのデータ

        #############################################################################
        """

        forecast_data = self.forecast_data
        next_time = int(current_time + time_step * 3600)
        target_TY = int(self.target_name)

        next_time_target_data = forecast_data.filter(
            (pl.col("unixtime") == next_time) & (pl.col("TYPHOON NUMBER") == target_TY)
        )

        return next_time_target_data

    # 状態量計算
    # 次の時刻での船の座標
    def get_next_position(self, time_step):
        """
        ############################ def get_next_position ############################

        [ 説明 ]

        台風発電船の次の時刻での座標を計算するための関数です。

        現在地から目標地点まで直線に進んだ場合にいる座標を計算して返します。

        台風発電船が次の時刻で目的地に到着できる場合は座標は目的地のものになります。

        状態量が更新されるのみなのでreturnでの戻り値はありません。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        target_position = (self.target_lat, self.target_lon)

        # 目的地と現在地の距離
        Goal_now_distance = self.get_distance(target_position)  # [km]

        # 船がtime_step時間で進める距離
        advance_distance = self.change_kt_kmh() * time_step

        # 緯度の差
        g_lat = self.target_lat
        n_lat = self.ship_lat

        lat_difference = g_lat - n_lat

        # 経度の差
        g_lon = self.target_lon
        n_lon = self.ship_lon

        lon_difference = g_lon - n_lon

        # 進める距離と目的地までの距離の比を出す
        if Goal_now_distance != 0:
            distance_ratio = advance_distance / Goal_now_distance
        else:
            distance_ratio = 0

        # 念の為の分岐
        # 距離の比が1を超える場合目的地に到着できることになるので座標を目的地へ、そうでないなら当該距離進める

        if distance_ratio < 1 and distance_ratio > 0:

            # 次の時間にいるであろう緯度
            next_lat = lat_difference * distance_ratio + n_lat

            # 次の時間にいるであろう経度
            next_lon = lon_difference * distance_ratio + n_lon

        else:

            # 次の時間にいるであろう緯度
            next_lat = g_lat

            # 次の時間にいるであろう経度
            next_lon = g_lon

        next_position = (next_lat, next_lon)
        self.ship_lat = next_lat
        self.ship_lon = next_lon

    def return_base_action(self, time_step):
        """
        ############################ def get_next_ship_state ############################

        [ 説明 ]

        台風発電船が拠点に帰港する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        self.target_lat = self.base_lat
        self.target_lon = self.base_lon

        self.go_base = 1
        self.brance_condition = "battery capacity exceeded specified ratio"

        # 帰港での船速の入力
        self.speed_kt = self.nomal_ave_speed

        self.target_name = "base station"

        base_ship_dis_time = (
            self.get_distance((self.base_lat, self.base_lon)) / self.change_kt_kmh()
        )
        # timestep後に拠点に船がついている場合
        if base_ship_dis_time <= time_step:
            self.brance_condition = "arrival at base station"
            self.go_base = 0
            self.TY_and_base_action = 0

            self.speed_kt = 0

            # 電気の積み下ろし
            self.supply_elect = self.storage - self.max_storage * 0.1
            self.storage = self.max_storage * 0.1

            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 0  # 0なら消費していない、1なら消費

            # 発電船状態入力
            self.ship_state = 0  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        else:
            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 1  # 0なら消費していない、1なら消費

            # 発電船状態入力
            self.ship_state = 4  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

    def return_standby_action(self, time_step):
        """
        ############################ def get_next_ship_state ############################

        [ 説明 ]

        台風発電船が待機位置に向かう場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        self.brance_condition = "returning to standby position as no typhoon"

        self.target_lat = self.standby_lat
        self.target_lon = self.standby_lon

        self.target_name = "Standby position"

        self.speed_kt = self.nomal_ave_speed
        standby_ship_dis_time = (
            self.get_distance((self.standby_lat, self.standby_lon))
            / self.change_kt_kmh()
        )

        if standby_ship_dis_time <= time_step:
            self.brance_condition = "arrival at standby position"

            self.speed_kt = 0

            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 0  # 0なら消費していない、1なら消費

            # 発電船状態入力
            self.ship_state = 0  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        else:
            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 1  # 0なら消費していない、1なら消費

            # 発電船状態入力
            self.ship_state = 4  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

    def typhoon_chase_action(self, time_step):
        """
        ############################ def typhoon_chase_action ############################

        [ 説明 ]

        台風発電船が台風を追従する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        追加で、拠点を経由するのかの判断も行います。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        self.speed_kt = self.max_speed

        max_speed_kmh = self.change_kt_kmh()

        # GS_dis_judge = TY_tracking_speed_kmh*self.distance_judge_hours

        TY_tracking_speed = (self.target_TY_data[0, "distance"]) / (
            self.target_TY_data[0, "TY_CATCH_TIME"]
        )  # その場から台風へ時間ぴったりに着くように移動する場合の船速

        # 算出したTY_tracking_speedが最大船速を超えないか判断。超える場合は最大船速に置き換え
        if TY_tracking_speed > max_speed_kmh:
            self.speed_kt = self.max_speed
        else:
            # km/hをktに変換
            self.speed_kt = TY_tracking_speed / 1.852

        # 追従対象の台風までの距離
        GS_TY_dis = self.target_TY_data[0, "distance"]

        self.brance_condition = "tracking typhoon at maximum ship speed started"

        self.target_lat = self.target_TY_data[0, "FORE_LAT"]
        self.target_lon = self.target_TY_data[0, "FORE_LON"]

        if self.target_TY_data[0, "TY_CATCH_TIME"] <= time_step:
            self.brance_condition = "arrived at typhoon"
            self.speed_kt = self.max_speed

            self.GS_gene_judge = 1

            self.GS_loss_judge = 0

            # 発電船状態入力
            self.ship_state = 1  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        else:

            self.brance_condition = "tracking typhoon"

            self.GS_gene_judge = 0

            self.GS_loss_judge = 1

            # 発電船状態入力
            self.ship_state = 2  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

            # 座標間距離を用いた発電の有無のチェック用数値
            self.distance_check = 1  # 1ならチェック必要

        # 拠点を経由できるか、するかの判断フェーズ
        direction_to_TY = self.get_direction((self.target_lat, self.target_lon))
        direction_to_base = self.get_direction((self.base_lat, self.base_lon))
        direction_difference = np.abs(direction_to_TY - direction_to_base)
        targetTY_base_dis = geodesic(
            (self.target_lat, self.target_lon), (self.base_lat, self.base_lon)
        ).km
        need_distance = (
            self.get_distance((self.base_lat, self.base_lon)) + targetTY_base_dis
        )
        max_speed_kmh = self.max_speed * 1.852
        need_time_hours = need_distance / max_speed_kmh
        TY_catch_time = self.target_TY_data[0, "TY_CATCH_TIME"]

        TY_distance = self.get_distance((self.target_lat, self.target_lon))
        base_distance = self.get_distance((self.base_lat, self.base_lon))
        distance_dis = TY_distance - base_distance

        self.TY_and_base_action = 0

        if self.storage_percentage >= self.sub_judge_energy_storage_per:
            if need_time_hours <= TY_catch_time:
                # 元の目的地に問題なくつけるのであれば即実行
                self.speed_kt = self.max_speed
                self.TY_and_base_action = (
                    1  # 台風に向かいながら拠点に帰港する行動のフラグ
                )

                self.return_base_action

                self.brance_condition = "tracking typhoon via base"

            else:
                if direction_difference < self.judge_direction and distance_dis > 0:
                    # 拠点の方が近くて、方位に大きな差がなければとりあえず経由する
                    self.speed_kt = self.max_speed
                    self.TY_and_base_action = (
                        1  # 台風に向かいながら拠点に帰港する行動のフラグ
                    )

                    self.return_base_action

                    self.brance_condition = "tracking typhoon via base"

    # 状態量計算
    # 行動判定も入っているので機能の要素も入っている？
    # 全てのパラメータを次の時刻のものに変える処理
    def get_next_ship_state(self, year, current_time, time_step):
        """
        ############################ def get_next_ship_state ############################

        [ 説明 ]

        台風発電船というエージェントの行動規則そのものの設定であるとともに各分岐条件での状態量の更新を行う関数です。

        行動規則は次のように設定されています。

        1.その時刻での蓄電量の割合がX％以上なら拠点へ帰還

        2.台風がない場合待機位置へ帰還

        3.台風が存在し、追いついている場合発電

        4.台風が存在し、追従中でかつそれが近い場合最大船速で追従

        5.台風が存在し、追従中でかつそれが遠い場合低速で追従

        以上の順番で台風発電船が置かれている状況を判断し、対応した行動を台風発電船がとるように設定している。

        そして、各行動に対応した状態量の更新を行なっている。

        ##############################################################################

        引数 :
            year (int) : シミュレーションを行う年
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        self.distance_check = 0

        # 蓄電量X％以上の場合
        if (
            self.storage_percentage >= self.judge_energy_storage_per
            or self.go_base == 1
        ):
            # if self.go_base == 1:
            self.speed_kt = self.nomal_ave_speed

            self.return_base_action(time_step)

            if self.standby_via_base == 1:
                self.brance_condition = "return standby via base"

            ############  ここでデータ取得から判断させるよりも台風発電の選択肢に行った時にフラグを立てる方が良いかも？  ###############

            # 追従対象の台風が存在するか判別
            self.target_TY_data = self.get_target_data(year, current_time, time_step)
            typhoon_num = len(self.target_TY_data)

            #############  近くに寄った場合に帰るという選択肢の追加  #####################

            # base_ship_dis = self.get_distance((self.base_lat,self.base_lon))

            if (
                typhoon_num == 0
                or self.storage_percentage >= self.judge_energy_storage_per
            ):  # 台風がないまたは蓄電容量規定値超え

                # 追従対象の台風がないことにする

                self.target_TY = 0

                self.next_TY_lat = 0
                self.next_TY_lon = 0
                self.next_ship_TY_dis = " "

            elif (
                self.storage_percentage >= self.sub_judge_energy_storage_per
            ):  # 少量の蓄電でも戻る場合の基準値を利用した場合

                if typhoon_num == 0:

                    # 追従対象の台風がないことにする

                    self.target_TY = 0

                    self.next_TY_lat = 0
                    self.next_TY_lon = 0
                    self.next_ship_TY_dis = " "

                elif self.TY_and_base_action == 1:

                    # 台風が来ているけど途中でよる場合の処理
                    self.brance_condition = "tracking typhoon via base"

                    # 最大船速でとっとと戻る
                    self.speed_kt = self.max_speed

                    self.target_name = str(self.target_TY_data[0, "TYPHOON NUMBER"])
                    self.target_TY = self.target_TY_data[0, "TYPHOON NUMBER"]

                    comparison_lat = self.target_TY_data[0, "FORE_LAT"]
                    comparison_lon = self.target_TY_data[0, "FORE_LON"]

                    next_time_TY_data = self.get_next_time_target_TY_data(
                        time_step, current_time
                    )

                    if len(next_time_TY_data) != 0:
                        self.next_TY_lat = next_time_TY_data[0, "FORE_LAT"]
                        self.next_TY_lon = next_time_TY_data[0, "FORE_LON"]
                        next_TY_locate = (self.next_TY_lat, self.next_TY_lon)

                        self.next_ship_TY_dis = self.get_distance(next_TY_locate)

                    else:
                        # 追従対象の台風がないことにする
                        self.next_TY_lat = 0
                        self.next_TY_lon = 0
                        self.next_ship_TY_dis = " "

                    if (
                        target_TY_lat != comparison_lat
                        or target_TY_lon != comparison_lon
                    ):

                        # 目標地点が変わりそうなら台風追従行動の方で再検討
                        self.typhoon_chase_action(time_step)

        # 蓄電量90％未満の場合
        else:
            standby_via_base = 0

            self.next_TY_lat = 0
            self.next_TY_lon = 0
            self.next_ship_TY_dis = " "

            self.speed_kt = self.max_speed
            # 追従対象の台風が存在するか判別
            self.target_TY_data = self.get_target_data(year, current_time, time_step)
            typhoon_num = len(self.target_TY_data)

            # 待機位置へ帰還
            if typhoon_num == 0:

                if self.storage_percentage >= self.sub_judge_energy_storage_per:
                    self.return_base_action(time_step)
                    self.brance_condition = "return standby via base"
                    self.standby_via_base = 1
                    self.target_TY = 0
                else:
                    self.return_standby_action(time_step)
                    self.target_TY = 0

            # 追従対象の台風が存在する場合
            elif typhoon_num >= 1:

                self.target_name = str(self.target_TY_data[0, "TYPHOON NUMBER"])
                self.target_TY = self.target_TY_data[0, "TYPHOON NUMBER"]

                next_time_TY_data = self.get_next_time_target_TY_data(
                    time_step, current_time
                )

                if len(next_time_TY_data) != 0:
                    self.next_TY_lat = next_time_TY_data[0, "FORE_LAT"]
                    self.next_TY_lon = next_time_TY_data[0, "FORE_LON"]
                    next_TY_locate = (self.next_TY_lat, self.next_TY_lon)

                    self.next_ship_TY_dis = self.get_distance(next_TY_locate)

                self.typhoon_chase_action(time_step)

                target_TY_lat = self.target_TY_data[0, "FORE_LAT"]
                target_TY_lon = self.target_TY_data[0, "FORE_LAT"]

                ####

                ########### 低速追従は考えないものとする ##############

                # elif target_TY_data[0,"TY_CATCH_TIME"] > judge_time and GS_TY_dis > GS_dis_judge:
                #    self.brance_condition = "typhoon is at a distance"

                #    self.speed_kt = self.max_speed

                #    GS_gene_judge = 0

                #    GS_loss_judge = 1

                #    self.brance_condition = "tracking typhoon at low speed from a distance"
                # 発電船状態入力
                #    self.ship_state = 2.5  #通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

                #    self.target_lat = target_TY_data[0,"FORE_LAT"]
                #    self.target_lon = target_TY_data[0,"FORE_LON"]

        # 次の時刻の発電船座標取得
        self.get_next_position(time_step)

        ##########現在時刻＋timestepの台風の座標を取得しておく##########
        # それを用いて台風の50km圏内に入っているかを考える分岐を作る
        if self.distance_check == 1:
            # next_time_TY_data = self.get_next_time_target_TY_data(time_step,current_time)

            self.distance_check = 0

            if len(next_time_TY_data) != 0:
                next_ship_locate = (self.ship_lat, self.ship_lon)

                self.next_TY_lat = next_time_TY_data[0, "FORE_LAT"]
                self.next_TY_lon = next_time_TY_data[0, "FORE_LON"]

                next_TY_locate = (self.next_TY_lat, self.next_TY_lon)

                self.next_ship_TY_dis = self.get_distance(next_TY_locate)

            if (
                len(next_time_TY_data) != 0
                and self.next_ship_TY_dis <= self.effective_range
            ):
                self.brance_condition = "within 50km of a typhoon following"

                self.GS_gene_judge = 1
                self.GS_loss_judge = 0

                self.ship_state = 1  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

            else:
                # self.brance_condition = "beyond 50km of a typhoon following"

                self.GS_gene_judge = 0
                self.GS_loss_judge = 1

                self.ship_state = 2  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        ##########################################################

        ############

        # 現在この関数での出力は次の時刻での　船の状態　追従目標　船速　座標　単位時間消費電力・発電量　保有電力　保有電力割合　目標地点との距離　となっている

        # その時刻〜次の時刻での消費電力計算
        self.loss_elect = (
            self.calculate_power_consumption(time_step) * self.GS_loss_judge
        )

        # その時刻〜次の時刻での発電量計算
        self.gene_elect = self.generator_output * time_step * self.GS_gene_judge

        self.total_gene_elect = self.total_gene_elect + self.gene_elect
        self.total_loss_elect = self.total_loss_elect + self.loss_elect

        self.total_gene_time = self.total_gene_time + time_step * self.GS_gene_judge
        self.total_loss_time = self.total_loss_time + time_step * self.GS_loss_judge

        # 次の時刻での発電船保有電力
        self.storage = self.storage + self.gene_elect - self.loss_elect

        self.storage_percentage = self.storage / self.max_storage * 100

        # 目標地点との距離
        target_position = (self.target_lat, self.target_lon)
        self.target_distance = self.get_distance(target_position)
