from datetime import datetime, timedelta, timezone

import polars as pl
from dateutil import tz
from tqdm import tqdm

from tpg_ship_sim.model import forecaster, tpg_ship


def get_TY_start_time(year, typhoon_path_forecaster):
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
        typhoon_path_forecaster (dataflame) : 過去の台風のデータ(unixtime追加後)

    戻り値 :
        TY_occurrence_time (list) : 各台風の発生時刻のリスト

    #############################################################################
    """

    TY_num = typhoon_path_forecaster.n_unique("TYPHOON NUMBER")

    # 台風発生時刻を入れておくリスト
    TY_occurrence_time = []

    # 各台風番号で開始時刻の取得
    for i in range(TY_num):
        TY_bangou = (year - 2000) * 100 + i + 1
        typhoon_data_by_num = typhoon_path_forecaster.filter(
            pl.col("TYPHOON NUMBER") == TY_bangou
        )
        typhoon_data_by_num = typhoon_data_by_num.select(
            pl.col("*").sort_by("unixtime")
        )
        TY_occurrence_time.append(typhoon_data_by_num[0, "unixtime"])

    return TY_occurrence_time


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


def simulate(
    tpg_ship_1,  # TPG ship
    typhoon_path_forecaster,  # Forecaster
    st_base,  # Storage base
    support_ship_1,  # Support ship 1
    support_ship_2,  # Support ship 2
    typhoon_data_path,
    tpg_ship_log_file_path,
    storage_base_log_file_path,
    support_ship_1_log_file_path,
    support_ship_2_log_file_path,
) -> None:

    year = 2019
    time_step = 6
    UTC = timezone(timedelta(hours=+0), "UTC")
    datetime_1_1 = datetime(year, 1, 1, 0, 0, 0, tzinfo=tz.gettz("UTC"))
    current_time = int(datetime_1_1.timestamp())
    month = datetime_1_1.month
    # 終了時刻
    datetime_12_31 = datetime(year, 12, 31, 18, 0, 0, tzinfo=tz.gettz("UTC"))
    unixtime_12_31 = int(datetime_12_31.timestamp())
    # unixtimeでの時間幅
    time_step_unix = 3600 * time_step

    # 繰り返しの回数
    record_count = int((unixtime_12_31 - current_time) / (time_step_unix) + 1)

    # 台風データ設定
    typhoon_path_forecaster.year = year
    # typhoon_data = pl.read_csv(
    #     "data/" + "typhoon_data_"
    #     # + str(int(time_step))
    #     # + "hour_intervals_verpl/table"
    #     + str(year) + "_" + str(int(time_step)) + "_interval.csv",
    #     # encoding="shift-jis",
    # )
    typhoon_data = pl.read_csv(typhoon_data_path)
    typhoon_path_forecaster.original_data = typhoon_data

    # 風データ設定
    wind_data = pl.read_csv(
        "data/wind_datas/era5_testdata_E180W90S0W90_"
        + str(int(year))
        + "_"
        + str(int(month))
        + ".csv"
    )

    # 発電船パラメータ設定
    tpg_ship_1.max_speed_power = cal_maxspeedpower(
        tpg_ship_1.max_speed,
        tpg_ship_1.max_storage,
        tpg_ship_1.storage_method,
        tpg_ship_1.hull_num,
    )  # 船体を最大船速で進めるための出力[W]

    tpg_ship_1.forecast_time = forecaster.Forecaster.forecast_time

    # 運搬船設定
    # support_ship.Support_ship.max_storage = tpg_ship_1.max_storage * 0.5
    # support_ship_1 = support_ship.Support_ship()
    # support_ship_2 = support_ship.Support_ship()

    # 拠点位置に関する設定
    # 発電船拠点位置
    tpg_ship_1.base_lat = st_base.locate[0]
    tpg_ship_1.base_lon = st_base.locate[1]

    tpg_ship_1.TY_start_time_list = get_TY_start_time(year, typhoon_data)
    # 待機位置に関する設定
    tpg_ship_1.standby_lat = st_base.locate[0]
    tpg_ship_1.standby_lon = st_base.locate[1]

    # tpg_ship_1.govia_base_judge_energy_storage_per = 20

    tpg_ship_1.set_initial_states()

    #####################################  出力用の設定  ############################################
    unix = []
    date = []

    ####################### TPG ship ##########################
    tpg_ship_1.set_outputs()

    ####################### Storage base ##########################
    st_base.set_outputs()

    ####################### Support ship ##########################
    support_ship_1.set_outputs()

    support_ship_2.set_outputs()

    #######################################  出力用リストへ入力  ###########################################
    unix.append(current_time)
    date.append(datetime.fromtimestamp(unix[-1], UTC))

    tpg_ship_1.outputs_append()
    GS_data = tpg_ship_1.get_outputs(unix, date)

    ####################### Storage base ##########################
    st_base.outputs_append()
    stBASE_data = st_base.get_outputs(unix, date)

    ####################### Support ship ##########################
    support_ship_1.outputs_append()
    support_ship_2.outputs_append()

    spSHIP1_data = support_ship_1.get_outputs(unix, date)
    spSHIP2_data = support_ship_2.get_outputs(unix, date)

    for data_num in tqdm(range(record_count), desc="Simulating..."):

        # 月毎の風データの取得
        if month != datetime.fromtimestamp(current_time, UTC).month:
            month = datetime.fromtimestamp(current_time, UTC).month
            wind_data = pl.read_csv(
                "data/wind_datas/era5_testdata_E180W90S0W90_"
                + str(int(year))
                + "_"
                + str(int(month))
                + ".csv"
            )

        # 予報データ取得
        tpg_ship_1.forecast_data = typhoon_path_forecaster.create_forecast(
            time_step, current_time
        )

        # timestep後の発電船の状態を取得
        tpg_ship_1.get_next_ship_state(year, current_time, time_step, wind_data)

        # timestep後の中継貯蔵拠点と運搬船の状態を取得
        st_base.operation_base(
            tpg_ship_1, support_ship_1, support_ship_2, year, current_time, time_step
        )

        # timestep後の時刻の取得
        current_time = current_time + time_step_unix

        #######################################  出力用リストへ入力  ###########################################
        unix.append(current_time)
        date.append(datetime.fromtimestamp(unix[-1], UTC))

        tpg_ship_1.outputs_append()
        GS_data = tpg_ship_1.get_outputs(unix, date)

        ####################### storageBASE ##########################
        st_base.outputs_append()
        stBASE_data = st_base.get_outputs(unix, date)

        ####################### supportSHIP ##########################
        support_ship_1.outputs_append()
        support_ship_2.outputs_append()

        spSHIP1_data = support_ship_1.get_outputs(unix, date)
        spSHIP2_data = support_ship_2.get_outputs(unix, date)

    GS_data.write_csv(tpg_ship_log_file_path)
    stBASE_data.write_csv(storage_base_log_file_path)
    spSHIP1_data.write_csv(support_ship_1_log_file_path)
    spSHIP2_data.write_csv(support_ship_2_log_file_path)


############################################################################################
