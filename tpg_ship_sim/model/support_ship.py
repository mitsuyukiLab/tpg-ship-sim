import numpy as np
from geopy.distance import geodesic


class Support_ship:
    """
    ############################### class support_ship ###############################

    [ 説明 ]

    このクラスは補助船を作成するクラスです。

    主にTPGshipが生成した電力や水素を貯蔵した中間貯蔵拠点から供給地点に輸送します。

    補助船の能力や状態量もここで定義されることになります。

    ##############################################################################

    引数 :
        year (int) : シミュレーションを行う年
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)
        storage_base_position (taple) : 中継貯蔵拠点の座標(緯度,経度)

    属性 :
        supplybase_lat (float) : 供給拠点の緯度
        supplybase_lon (float) : 供給拠点の経度
        ship_lat (float) : 補助船のその時刻での緯度
        ship_lon (float) : 補助船のその時刻での経度
        max_storage (float) : 中継貯蔵拠点の蓄電容量の上限値
        support_ship_speed (float) : 補助船の最大船速
        storage (float) : 中継貯蔵拠点のその時刻での蓄電量
        ship_gene (int) : 補助船が発生したかどうかのフラグ
        arrived_supplybase (int) : 供給拠点に到着したかどうかのフラグ
        arrived_storagebase (int) : 中継貯蔵拠点に到着したかどうかのフラグ
        target_lat (float) : 補助船の目標地点の緯度
        target_lon (float) : 補助船の目標地点の経度
        brance_condition (str) : 補助船の行動の記録

    """

    ship_gene = 0
    storage = 0
    arrived_supplybase = 1
    arrived_storagebase = 0

    target_lat = np.nan
    target_lon = np.nan
    brance_condition = "no action"

    def __init__(self, supply_base_locate, max_storage_wh, max_speed_kt) -> None:
        self.supplybase_lat = supply_base_locate[0]
        self.supplybase_lon = supply_base_locate[1]
        self.ship_lat = supply_base_locate[0]
        self.ship_lon = supply_base_locate[1]
        self.max_storage = max_storage_wh
        self.support_ship_speed = max_speed_kt

    # 状態量計算
    def get_distance(self, storage_base_position):
        """
        ############################ def get_distance ############################

        [ 説明 ]

        補助船(または拠点位置or観測地点)からUUVへの距離を計算する関数です。

        ##############################################################################

        引数 :
            storage_base_position (taple) : 中継貯蔵拠点の座標(緯度,経度)


        戻り値 :
            distance (float) : 補助船から上記拠点への距離(km)

        #############################################################################
        """

        A_position = (self.ship_lat, self.ship_lon)

        # AーB間距離
        distance = geodesic(A_position, storage_base_position).km

        return distance

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

    # 拠点または観測地点からuuvまでの距離取得
    def get_base_dis_data(self, storage_base_position):
        """
        ############################ def get_base_dis_data ############################

        [ 説明 ]

        get_target_dataで選ばれ、追従対象となった台風のcurrent_time + time_stepの時刻での座標を取得する関数です。

        存在しない場合は空のデータフレームが返ります。

        本関数は複数の供給拠点が本土にある場合に初めて有効となる。今後の拡張を見越し、事前実装する。

        課題点としては、各拠点で運用される補助船がおそらく共通化してしまうことである。

        コスト計算をするなら拠点ごとに設定する必要がある。

        ##############################################################################

        引数 :
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]
            storage_base_position (taple) : 中継貯蔵拠点の座標(緯度,経度)


        戻り値 :
            self.route_plus_dis_data (dataflame) : 供給拠点または中継貯蔵拠点からの距離が追加されたデータ

        #############################################################################
        """
        distance_list = []

        if len(self.base_data) != 0:
            for i in range(len(self.base_data)):

                A_position = (self.base_data[i, "LAT"], self.base_data[i, "LON"])
                distance = geodesic(A_position, storage_base_position).km
                distance_list.append(distance)

            # 台風の距離を一応書いておく
            base_plus_dis_data = self.base_data.with_columns(
                pl.Series(distance_list).alias("distance")
            )

            # 距離が近い順番に並び替え
            base_plus_dis_data = base_plus_dis_data.select(
                pl.col("*").sort_by("distance", descending=False)
            )

        return base_plus_dis_data

    # 状態量計算
    # 次の時刻での船の座標
    def get_next_position(self, time_step):
        """
        ############################ def get_next_position ############################

        [ 説明 ]

        補助船の次の時刻での座標を計算するための関数です。

        現在地から目標地点まで直線に進んだ場合にいる座標を計算して返します。

        補助船が次の時刻で目的地に到着できる場合は座標は目的地のものになります。

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

    # まだ使わないver5では拠点は1つ
    def set_start_position(self, storage_base_position):

        self.change_ship = 0
        self.base_plus_dis_data = self.get_base_dis_data(storage_base_position)
        if not np.isnan(self.ship_lat):
            uuv_ship_dis = self.get_distance(storage_base_position)
            if uuv_ship_dis > self.base_plus_dis_data[0, "distance"]:
                self.change_ship = 1
                # self.ship_lat = self.route_plus_dis_data[0,"LAT"]
                # self.ship_lon = self.route_plus_dis_data[0,"LON"]
                # self.base_lat = self.route_plus_dis_data[0,"LAT"]
                # self.base_lon = self.route_plus_dis_data[0,"LON"]
            else:
                # 更新なし、その場からuuvへ向かう
                self.ship_gene = 1
                self.arrived_supplybase = 0
                self.arrived_storagebase = 0

                self.target_lat = storage_base_position[0]
                self.target_lon = storage_base_position[1]

        else:
            self.ship_lat = self.base_plus_dis_data[0, "LAT"]
            self.ship_lon = self.base_plus_dis_data[0, "LON"]
            self.base_lat = self.base_plus_dis_data[0, "LAT"]
            self.base_lon = self.base_plus_dis_data[0, "LON"]
            self.arrived_supplybase = 0
            self.arrived_storagebase = 0
            self.ship_gene = 1

            self.target_lat = storage_base_position[0]
            self.target_lon = storage_base_position[1]

    def go_storagebase_action(self, storage_base_position, time_step):
        """
        ############################ def get_next_ship_state ############################

        [ 説明 ]

        UUVが拠点に帰港する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        # 帰港での船速の入力
        self.speed_kt = self.support_ship_speed

        self.target_lat = storage_base_position[0]
        self.target_lon = storage_base_position[1]

        storagebase_ship_dis_time = (
            self.get_distance(storage_base_position) / self.change_kt_kmh()
        )
        self.arrived_storagebase = 0
        self.arrived_supplybase = 0

        # timestep後にUUVに船がついている場合
        if storagebase_ship_dis_time <= time_step:
            self.brance_condition = "arrival at storage Base"
            self.arrived_storagebase = 1  # 到着のフラグ
            self.arrived_supplybase = 0
            self.ship_lat = storage_base_position[0]
            self.ship_lon = storage_base_position[1]

            self.speed_kt = 0

        else:
            self.brance_condition = "go to storage Base"
            self.arrived_storagebase = 0

    def go_supplybase_action(self, time_step):
        """
        ############################ def go_supplybase_action ############################

        [ 説明 ]

        補助船が供給拠点に帰港する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        # 帰港での船速の入力
        self.speed_kt = self.support_ship_speed
        self.target_lat = self.supplybase_lat
        self.target_lon = self.supplybase_lon
        supplybase_position = (self.supplybase_lat, self.supplybase_lon)
        uuv_ship_dis_time = (
            self.get_distance(supplybase_position) / self.change_kt_kmh()
        )
        self.arrived_supplybase = 0

        # timestep後にBaseに船がついている場合
        if uuv_ship_dis_time <= time_step:
            self.brance_condition = "arrival at supply Base"
            self.ship_gene = 0
            self.arrived_supplybase = 1  # 到着のフラグ
            self.arrived_storagebase = 0  # 履歴削除
            self.ship_lat = self.supplybase_lat
            self.ship_lon = self.supplybase_lon
            self.target_lat = np.nan
            self.target_lon = np.nan
            self.target_distance = np.nan
            self.storage = 0

            self.speed_kt = 0

        else:
            self.brance_condition = "go to supply Base"
            self.arrived_supplybase = 0

    def get_next_ship_state(self, storage_base_position, year, current_time, time_step):

        if (self.arrived_storagebase == 0) and (self.arrived_supplybase == 1):
            self.go_storagebase_action(storage_base_position, time_step)
        elif (self.arrived_storagebase == 1) and (self.arrived_supplybase == 0):
            self.go_supplybase_action(time_step)
        elif (self.arrived_storagebase == 0) and (self.arrived_supplybase == 0):
            self.go_storagebase_action(storage_base_position, time_step)
        else:
            self.arrived_supplybase = 0
            self.arrived_storagebase = 0

        # 拠点に帰った場合を弾く
        if not np.isnan(self.target_lat):
            self.get_next_position(time_step)

            # 目標地点との距離
            target_position = (self.target_lat, self.target_lon)
            self.target_distance = self.get_distance(target_position)
