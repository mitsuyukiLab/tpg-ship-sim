from datetime import datetime, timedelta, timezone

import polars as pl
from dateutil import tz
from tqdm import tqdm

from tpg_ship_sim.model import forecaster, storage_base, support_ship, tpg_ship


def get_TY_start_time(year, TY_data):
    """
    ############################## def get_TY_start_time ##############################

    [ 説明 ]

    この関数は台風の発生時刻を取得するための関数です。

    本来は発生したごとに逐次記録すれば良いのですが、そのプログラムを作っても嵩張るだけだと思ったので、

    予報期間に関係なく発生時間は取得できるものとしてリスト化することにしました。

    add_unixtimeで処理したデータが必要です。

    ##############################################################################

    引数 :
        year (int) : シミュレーションを行う年
        TY_data (dataflame) : 過去の台風のデータ(unixtime追加後)

    戻り値 :
        TY_occurrence_time (list) : 各台風の発生時刻のリスト

    #############################################################################
    """

    TY_num = TY_data.n_unique("TYPHOON NUMBER")

    # 台風発生時刻を入れておくリスト
    TY_occurrence_time = []

    # 各台風番号で開始時刻の取得
    for i in range(TY_num):
        TY_bangou = (year - 2000) * 100 + i + 1
        typhoon_data_by_num = TY_data.filter(pl.col("TYPHOON NUMBER") == TY_bangou)
        typhoon_data_by_num = typhoon_data_by_num.select(
            pl.col("*").sort_by("unixtime")
        )
        TY_occurrence_time.append(typhoon_data_by_num[0, "unixtime"])

    return TY_occurrence_time


# 蓄電量の状態量のタイプわけ
def get_storage_state(storage_percentage):
    """
    ############################## def get_TY_start_time ##############################

    [ 説明 ]

    この関数は台風発電船の蓄電割合から対応する数値を返す関数です。

    この数値はシミュレーションの可視化の際に使われる数値です。

    ##############################################################################

    引数 :
        storage_percentage (float) : 台風発電船の蓄電割合

    戻り値 :
        20%以下→1 , 20%より多く80%より少ない→2 , 80%以上→3 , 100%以上→4

    #############################################################################
    """

    # 蓄電量が20％以下
    if storage_percentage <= 20:

        return 1
    # 蓄電量が100％以上
    elif storage_percentage >= 100:

        return 4
    # 蓄電量が80％以上
    elif storage_percentage >= 80:

        return 3
    # 蓄電量が20％より多く、80％より少ない
    else:

        return 2


def cal_dwt(storage, storage_method):
    # 載貨重量トンを算出する。単位はt。

    if storage_method == 1:  # 電気貯蔵
        # 重量エネルギー密度1000Wh/kgの電池を使うこととする。
        dwt = storage / 1000 / 1000

    elif storage_method == 2:  # 水素貯蔵
        # 有機ハイドライドで水素を貯蔵することとする。
        dwt = storage / 5000 * 0.0898 / 47.4

    else:
        print("cannot cal")

    return dwt


def cal_maxspeedpower(max_speed, storage, storage_method, body_num):

    dwt = cal_dwt(storage, storage_method) / body_num

    if storage_method == 1:  # 電気貯蔵
        # バルカー型
        k = 1.7
        power = k * (dwt ** (2 / 3)) * (max_speed**3) * body_num

    elif storage_method == 2:  # 水素貯蔵
        # タンカー型
        k = 2.2
        power = k * (dwt ** (2 / 3)) * (max_speed**3) * body_num

    else:
        print("cannot cal")

    return power


############################################################################################


def simulate(typhoon_data_path, output_folder_path) -> None:
    year = 2019
    time_step = 6
    UTC = timezone(timedelta(hours=+0), "UTC")
    datetime_1_1 = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz.gettz("UTC"))
    current_time = int(datetime_1_1.timestamp())
    # 終了時刻
    datetime_12_31 = datetime(year, 12, 31, 18, 0, 0, tzinfo=tz.gettz("UTC"))
    unixtime_12_31 = int(datetime_12_31.timestamp())
    # unixtimeでの時間幅
    time_step_unix = 3600 * time_step

    # 繰り返しの回数
    record_count = int((unixtime_12_31 - current_time) / (time_step_unix) + 1)

    # 台風データ設定
    forecaster.Forecaster.forecast_time = 24 * 5
    forecaster.Forecaster.slope = 0.0
    TY_data = forecaster.Forecaster()
    TY_data.year = year
    # typhoon_data = pl.read_csv(
    #     "data/" + "typhoon_data_"
    #     # + str(int(time_step))
    #     # + "hour_intervals_verpl/table"
    #     + str(year) + "_" + str(int(time_step)) + "_interval.csv",
    #     # encoding="shift-jis",
    # )
    typhoon_data = pl.read_csv(typhoon_data_path)
    TY_data.original_data = typhoon_data

    # 発電船パラメータ設定

    max_speed_kt = 20
    tpg_ship.TPGship.max_storage = 70 * (10**9)  # 蓄電容量[Wh]
    tpg_ship.TPGship.generator_output = 0.138 * (10**9)  # 定格出力[W]
    tpg_ship.TPGship.max_speed_power = cal_maxspeedpower(
        max_speed_kt, tpg_ship.TPGship.max_storage, 2, 1
    )  # 船体を最大船速で進めるための出力[W]
    tpg_ship.TPGship.generating_facilities_need_max_power = (
        tpg_ship.TPGship.generator_output * 0.01
    )  # 発電付加物分抵抗[W] (今回は定格出力の1％が停止状態での抵抗)
    tpg_ship.TPGship.wind_propulsion_power = (
        tpg_ship.TPGship.max_speed_power * 0.1
    )  # 風力推進機による推進力[W]

    ship1 = tpg_ship.TPGship()
    ship1.forecast_time = forecaster.Forecaster.forecast_time

    # 中継貯蔵拠点設定
    storage_base.storage_BASE.max_storage = ship1.max_storage * 3
    st_base = storage_base.storage_BASE()

    # 運搬船設定
    support_ship.support_SHIP.max_storage = ship1.max_storage * 0.5
    supportSHIP1 = support_ship.support_SHIP()
    supportSHIP2 = support_ship.support_SHIP()

    # 拠点位置に関する設定
    # 発電船拠点位置
    ship1.base_lat = st_base.lat
    ship1.base_lon = st_base.lon

    ship1.TY_start_time_list = get_TY_start_time(year, typhoon_data)
    # 待機位置に関する設定
    ship1.standby_lat = st_base.lat
    ship1.standby_lon = st_base.lon

    # ship1.sub_judge_energy_storage_per = 20

    ship1.set_initial_states()

    # 外部初期値入力
    storage_state_num = get_storage_state(ship1.storage_percentage)

    #####################################  出力用の設定  ############################################
    # 発電船の行動詳細
    branch_condition_list = []

    # 台風の番号
    target_typhoon_num = []  # そのときに追従している台風の番号（ない場合は0が入る）

    # 目標地点
    target_name_list = []
    target_lat_list = []
    target_lon_list = []
    target_dis_list = []

    # 台風座標
    TY_lat_list = []
    TY_lon_list = []

    # 発電船台風間距離
    GS_TY_dis_list = []

    # 発電船の座標
    GS_lat_list = []
    GS_lon_list = []

    # 時刻関係
    unix = []  # unixtime
    date = []  # datetime

    # 発電船の状態
    GS_state_list = []  # 発電船の行動状態(描画用数値)
    GS_speed_list = []

    ############################# 発電指数 ###############################
    GS_elect_storage_percentage = []  # 船内蓄電割合
    GS_storage_state = []
    gene_elect_time = []  # 発電時間
    total_gene_elect = []  # 総発電量
    loss_elect_time = []  # 電力消費時間（航行時間）
    total_loss_elect = []  # 総消費電力
    balance_gene_elect = []  # 発電収支（船内蓄電量）
    per_timestep_gene_elect = []  # 時間幅あたりの発電量
    per_timestep_loss_elect = []  # 時間幅あたりの消費電力
    year_round_balance_gene_elect = []  # 通年発電収支

    ####################### storageBASE ##########################
    stbase_storage = []
    stbase_st_per = []
    stbase_condition = []

    ####################### supportSHIP ##########################
    sp_target_lat1 = []
    sp_target_lon1 = []
    sp_storage1 = []
    sp_st_per1 = []
    sp_ship_lat1 = []
    sp_ship_lon1 = []
    sp_brance_condition1 = []

    sp_target_lat2 = []
    sp_target_lon2 = []
    sp_storage2 = []
    sp_st_per2 = []
    sp_ship_lat2 = []
    sp_ship_lon2 = []
    sp_brance_condition2 = []

    #######################################  出力用リストへ入力  ###########################################

    branch_condition_list.append(ship1.brance_condition)
    unix.append(current_time)
    date.append(datetime.fromtimestamp(unix[-1], UTC))

    target_name_list.append(ship1.target_name)
    target_lat_list.append(ship1.target_lat)
    target_lon_list.append(ship1.target_lon)
    target_dis_list.append(ship1.target_distance)

    target_typhoon_num.append(ship1.target_TY)
    TY_lat_list.append(ship1.next_TY_lat)
    TY_lon_list.append(ship1.next_TY_lon)
    GS_TY_dis_list.append(ship1.next_ship_TY_dis)

    GS_lat_list.append(ship1.ship_lat)
    GS_lon_list.append(ship1.ship_lon)
    GS_state_list.append(ship1.ship_state)
    GS_speed_list.append(ship1.speed_kt)

    per_timestep_gene_elect.append(ship1.gene_elect)  # 時間幅あたりの発電量[Wh]
    gene_elect_time.append(ship1.total_gene_time)  # 発電時間[hour]
    total_gene_elect.append(ship1.total_gene_elect)  # 総発電量[Wh]

    per_timestep_loss_elect.append(ship1.loss_elect)  # 時間幅あたりの消費電力[Wh]
    loss_elect_time.append(ship1.total_loss_time)  # 電力消費時間（航行時間）[hour]
    total_loss_elect.append(ship1.total_loss_elect)  # 総消費電力[Wh]

    ship1.storage_percentage = (ship1.storage / ship1.max_storage) * 100
    ship1.storage_state = get_storage_state(ship1.storage_percentage)
    GS_elect_storage_percentage.append(ship1.storage_percentage)  # 船内蓄電割合[%]
    GS_storage_state.append(ship1.storage_state)

    balance_gene_elect.append(ship1.storage)  # 発電収支（船内蓄電量）[Wh]

    year_round_balance_gene_elect.append(
        ship1.total_gene_elect - ship1.total_loss_elect
    )  # 通年発電収支

    GS_data = pl.DataFrame(
        {
            "unixtime": unix,
            "datetime": date,
            "TARGET LOCATION": target_name_list,
            "TARGET LAT": target_lat_list,
            "TARGET LON": target_lon_list,
            "TARGET DISTANCE[km]": target_dis_list,
            "TARGET TYPHOON": target_typhoon_num,
            "TARGET TY LAT": TY_lat_list,
            "TARGET TY LON": TY_lon_list,
            "TPGSHIP LAT": GS_lat_list,
            "TPGSHIP LON": GS_lon_list,
            "TPG_TY DISTANCE[km]": GS_TY_dis_list,
            "BRANCH CONDITION": branch_condition_list,
            "TPGSHIP STATUS": GS_state_list,
            "SHIP SPEED[kt]": GS_speed_list,
            "TIMESTEP POWER GENERATION[Wh]": per_timestep_gene_elect,
            "TOTAL GENE TIME[h]": gene_elect_time,
            "TOTAL POWER GENERATION[Wh]": total_gene_elect,
            "TIMESTEP POWER CONSUMPTION[Wh]": per_timestep_loss_elect,
            "TOTAL CONS TIME[h]": loss_elect_time,
            "TOTAL POWER CONSUMPTION[Wh]": total_loss_elect,
            "ONBOARD POWER STORAGE PER[%]": GS_elect_storage_percentage,
            "ONBOARD POWER STORAGE STATUS": GS_storage_state,
            "ONBOARD ENERGY STORAGE[Wh]": balance_gene_elect,
            "YEARLY POWER GENERATION BALANCE": year_round_balance_gene_elect,
        }
    )

    ####################### storageBASE ##########################
    stbase_storage.append(st_base.storage)
    stbase_st_per.append(st_base.storage / st_base.max_storage * 100)
    stbase_condition.append(st_base.brance_condition)

    stBASE_data = pl.DataFrame(
        {
            "unixtime": unix,
            "datetime": date,
            "STORAGE[Wh]": stbase_storage,
            "STORAGE PER[%]": stbase_st_per,
            "BRANCH CONDITION": stbase_condition,
        }
    )

    ####################### supportSHIP ##########################
    sp_target_lat1.append(supportSHIP1.target_lat)
    sp_target_lon1.append(supportSHIP1.target_lon)
    sp_storage1.append(supportSHIP1.storage)
    sp_st_per1.append(supportSHIP1.storage / supportSHIP1.max_storage * 100)
    sp_ship_lat1.append(supportSHIP1.ship_lat)
    sp_ship_lon1.append(supportSHIP1.ship_lon)
    sp_brance_condition1.append(supportSHIP1.brance_condition)

    sp_target_lat2.append(supportSHIP2.target_lat)
    sp_target_lon2.append(supportSHIP2.target_lon)
    sp_storage2.append(supportSHIP2.storage)
    sp_st_per2.append(supportSHIP2.storage / supportSHIP2.max_storage * 100)
    sp_ship_lat2.append(supportSHIP2.ship_lat)
    sp_ship_lon2.append(supportSHIP2.ship_lon)
    sp_brance_condition2.append(supportSHIP2.brance_condition)

    spSHIP1_data = pl.DataFrame(
        {
            "unixtime": unix,
            "datetime": date,
            "targetLAT": sp_target_lat1,
            "targetLON": sp_target_lon1,
            "LAT": sp_ship_lat1,
            "LON": sp_ship_lon1,
            "STORAGE[Wh]": sp_storage1,
            "STORAGE PER[%]": sp_st_per1,
            "BRANCH CONDITION": sp_brance_condition1,
        }
    )
    spSHIP2_data = pl.DataFrame(
        {
            "unixtime": unix,
            "datetime": date,
            "targetLAT": sp_target_lat2,
            "targetLON": sp_target_lon2,
            "LAT": sp_ship_lat2,
            "LON": sp_ship_lon2,
            "STORAGE[Wh]": sp_storage2,
            "STORAGE PER[%]": sp_st_per2,
            "BRANCH CONDITION": sp_brance_condition2,
        }
    )

    for data_num in tqdm(range(record_count), desc="Simulating..."):

        # 予報データ取得
        ship1.forecast_data = TY_data.create_forecast(time_step, current_time)

        # timestep後の発電船の状態を取得
        ship1.get_next_ship_state(year, current_time, time_step)

        # timestep後の中継貯蔵拠点と運搬船の状態を取得
        st_base.operation_base(
            ship1, supportSHIP1, supportSHIP2, year, current_time, time_step
        )

        # timestep後の時刻の取得
        current_time = current_time + time_step_unix

        #######################################  出力用リストへ入力  ###########################################

        branch_condition_list.append(ship1.brance_condition)
        unix.append(current_time)
        date.append(datetime.fromtimestamp(unix[-1], UTC))

        target_name_list.append(ship1.target_name)
        target_lat_list.append(ship1.target_lat)
        target_lon_list.append(ship1.target_lon)
        target_dis_list.append(ship1.target_distance)

        target_typhoon_num.append(ship1.target_TY)
        TY_lat_list.append(ship1.next_TY_lat)
        TY_lon_list.append(ship1.next_TY_lon)
        GS_TY_dis_list.append(ship1.next_ship_TY_dis)

        GS_lat_list.append(ship1.ship_lat)
        GS_lon_list.append(ship1.ship_lon)
        GS_state_list.append(ship1.ship_state)
        GS_speed_list.append(ship1.speed_kt)

        per_timestep_gene_elect.append(ship1.gene_elect)  # 時間幅あたりの発電量[Wh]
        gene_elect_time.append(ship1.total_gene_time)  # 発電時間[hour]
        total_gene_elect.append(ship1.total_gene_elect)  # 総発電量[Wh]

        per_timestep_loss_elect.append(ship1.loss_elect)  # 時間幅あたりの消費電力[Wh]
        loss_elect_time.append(ship1.total_loss_time)  # 電力消費時間（航行時間）[hour]
        total_loss_elect.append(ship1.total_loss_elect)  # 総消費電力[Wh]

        ship1.storage_percentage = (ship1.storage / ship1.max_storage) * 100
        ship1.storage_state = get_storage_state(ship1.storage_percentage)
        GS_elect_storage_percentage.append(ship1.storage_percentage)  # 船内蓄電割合[%]
        GS_storage_state.append(ship1.storage_state)

        balance_gene_elect.append(ship1.storage)  # 発電収支（船内蓄電量）[Wh]

        year_round_balance_gene_elect.append(
            ship1.total_gene_elect - ship1.total_loss_elect
        )  # 通年発電収支

        GS_data = pl.DataFrame(
            {
                "unixtime": unix,
                "datetime": date,
                "TARGET LOCATION": target_name_list,
                "TARGET LAT": target_lat_list,
                "TARGET LON": target_lon_list,
                "TARGET DISTANCE[km]": target_dis_list,
                "TARGET TYPHOON": target_typhoon_num,
                "TARGET TY LAT": TY_lat_list,
                "TARGET TY LON": TY_lon_list,
                "TPGSHIP LAT": GS_lat_list,
                "TPGSHIP LON": GS_lon_list,
                "TPG_TY DISTANCE[km]": GS_TY_dis_list,
                "BRANCH CONDITION": branch_condition_list,
                "TPGSHIP STATUS": GS_state_list,
                "SHIP SPEED[kt]": GS_speed_list,
                "TIMESTEP POWER GENERATION[Wh]": per_timestep_gene_elect,
                "TOTAL GENE TIME[h]": gene_elect_time,
                "TOTAL POWER GENERATION[Wh]": total_gene_elect,
                "TIMESTEP POWER CONSUMPTION[Wh]": per_timestep_loss_elect,
                "TOTAL CONS TIME[h]": loss_elect_time,
                "TOTAL POWER CONSUMPTION[Wh]": total_loss_elect,
                "ONBOARD POWER STORAGE PER[%]": GS_elect_storage_percentage,
                "ONBOARD POWER STORAGE STATUS": GS_storage_state,
                "ONBOARD ENERGY STORAGE[Wh]": balance_gene_elect,
                "YEARLY POWER GENERATION BALANCE": year_round_balance_gene_elect,
            }
        )

        ####################### storageBASE ##########################
        stbase_storage.append(st_base.storage)
        stbase_st_per.append(st_base.storage / st_base.max_storage * 100)
        stbase_condition.append(st_base.brance_condition)

        stBASE_data = pl.DataFrame(
            {
                "unixtime": unix,
                "datetime": date,
                "STORAGE[Wh]": stbase_storage,
                "STORAGE PER[%]": stbase_st_per,
                "BRANCH CONDITION": stbase_condition,
            }
        )

        ####################### supportSHIP ##########################
        sp_target_lat1.append(supportSHIP1.target_lat)
        sp_target_lon1.append(supportSHIP1.target_lon)
        sp_storage1.append(supportSHIP1.storage)
        sp_st_per1.append(supportSHIP1.storage / supportSHIP1.max_storage * 100)
        sp_ship_lat1.append(supportSHIP1.ship_lat)
        sp_ship_lon1.append(supportSHIP1.ship_lon)
        sp_brance_condition1.append(supportSHIP1.brance_condition)

        sp_target_lat2.append(supportSHIP2.target_lat)
        sp_target_lon2.append(supportSHIP2.target_lon)
        sp_storage2.append(supportSHIP2.storage)
        sp_st_per2.append(supportSHIP2.storage / supportSHIP2.max_storage * 100)
        sp_ship_lat2.append(supportSHIP2.ship_lat)
        sp_ship_lon2.append(supportSHIP2.ship_lon)
        sp_brance_condition2.append(supportSHIP2.brance_condition)

        spSHIP1_data = pl.DataFrame(
            {
                "unixtime": unix,
                "datetime": date,
                "targetLAT": sp_target_lat1,
                "targetLON": sp_target_lon1,
                "LAT": sp_ship_lat1,
                "LON": sp_ship_lon1,
                "STORAGE[Wh]": sp_storage1,
                "STORAGE PER[%]": sp_st_per1,
                "BRANCH CONDITION": sp_brance_condition1,
            }
        )
        spSHIP2_data = pl.DataFrame(
            {
                "unixtime": unix,
                "datetime": date,
                "targetLAT": sp_target_lat2,
                "targetLON": sp_target_lon2,
                "LAT": sp_ship_lat2,
                "LON": sp_ship_lon2,
                "STORAGE[Wh]": sp_storage2,
                "STORAGE PER[%]": sp_st_per2,
                "BRANCH CONDITION": sp_brance_condition2,
            }
        )

    GS_data.write_csv(output_folder_path + "/tpg_ship_1.csv")
    stBASE_data.write_csv(output_folder_path + "/storage_base_1.csv")
    spSHIP1_data.write_csv(output_folder_path + "/support_ship_1.csv")
    spSHIP2_data.write_csv(output_folder_path + "/support_ship_2.csv")


############################################################################################
