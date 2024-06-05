import polars as pl


class Storage_base:
    """
    ############################### class storage_base ###############################

    [ 説明 ]

    このクラスは中継貯蔵拠点を作成するクラスです。

    主にTPGshipが生成した電力や水素を貯蔵し、補助船に渡します。

    中継貯蔵拠点の能力や状態量もここで定義されることになります。

    ##############################################################################

    引数 :
        year (int) : シミュレーションを行う年
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)
        support_ship_1 (class) : Support_shipクラスのインスタンスその1
        support_ship_2 (class) : Support_shipクラスのインスタンスその2
        TPGship1 (class) : TPGshipクラスのインスタンスその1

    属性 :
        max_storage (float) : 中継貯蔵拠点の蓄電容量の上限値
        storage (float) : 中継貯蔵拠点のその時刻での蓄電量
        call_num (int) : supportSHIPを読んだ回数
        call_ship1 (int) : support_ship_1を呼ぶフラグ
        call_ship2 (int) : support_ship_1を呼ぶフラグ
        call_per (int) : supprotSHIPを呼ぶ貯蔵パーセンテージ

    """

    ####################################  パラメータ  ######################################

    max_storage = 0
    storage = 0
    call_num = 0
    call_ship1 = 0
    call_ship2 = 0
    call_per = 60
    brance_condition = "while in storage"
    MCH_discharged_cargo_quantity = 0

    def __init__(self, locate, max_storage) -> None:
        self.locate = locate
        self.max_storage = max_storage

    def set_outputs(self):
        """
        ############################ def set_outputs ############################

        [ 説明 ]

        中継貯蔵拠点の出力を記録するリストを作成する関数です。

        ##############################################################################

        """
        self.stbase_storage_list = []
        self.stbase_MCH_discharged_cargo_quantity_list = []
        self.stbase_st_per_list = []
        self.stbase_condition_list = []

    def outputs_append(self):
        """
        ############################ def outputs_append ############################

        [ 説明 ]

        set_outputs関数で作成したリストに出力を記録する関数です。

        ##############################################################################

        """
        self.stbase_storage_list.append(float(self.storage))
        self.stbase_MCH_discharged_cargo_quantity_list.append(
            float(self.MCH_discharged_cargo_quantity)
        )
        self.stbase_st_per_list.append(float(self.storage / self.max_storage * 100))
        self.stbase_condition_list.append(self.brance_condition)

    def get_outputs(self, unix, date):
        """
        ############################ def get_outputs ############################

        [ 説明 ]

        set_outputs関数で作成したリストを出力する関数です。

        ##############################################################################

        """
        data = pl.DataFrame(
            {
                "unixtime": unix,
                "datetime": date,
                "STORAGE[Wh]": self.stbase_storage_list,
                "MCH DISCHARGED CARGO QUANTITY[kg]": self.stbase_MCH_discharged_cargo_quantity_list,
                "STORAGE PER[%]": self.stbase_st_per_list,
                "BRANCH CONDITION": self.stbase_condition_list,
            }
        )

        return data

    ####################################  メソッド  ######################################

    def storage_elect(self, TPGship1):
        """
        ############################ def storage_elect ############################

        [ 説明 ]

        中継貯蔵拠点がTPGshipから発電成果物を受け取り、蓄電容量を更新する関数です。

        TPGshipが拠点に帰港した時にのみ増加する。それ以外は出力の都合で、常に0を受け取っている。

        """

        self.storage = self.storage + TPGship1.supply_elect
        self.MCH_discharged_cargo_quantity = (
            self.MCH_discharged_cargo_quantity + TPGship1.supply_elect
        )

        if TPGship1.supply_elect > 0:
            TPGship1.supply_elect = 0

    def supply_elect(
        self, support_ship_1, support_ship_2, year, current_time, time_step
    ):
        """
        ############################ def supply_elect ############################

        [ 説明 ]

        中継貯蔵拠点がsupportSHIPを呼び出し、貯蔵しているエネルギーを渡す関数。

        呼ぶまでの関数なので、読んだ後のsupportSHIPが帰るフェーズは別で記載している。

        """
        if (support_ship_1.arrived_supplybase == 1) or (
            self.call_ship1 == 1
        ):  # support_ship_1が活動可能な場合
            self.brance_condition = "call ship1"
            self.call_ship1 = 1
            support_ship_1.get_next_ship_state(
                self.locate, year, current_time, time_step
            )

            if support_ship_1.arrived_storagebase == 1:
                self.call_ship1 = 0
                if self.storage <= support_ship_1.max_storage:
                    support_ship_1.storage = support_ship_1.storage + self.storage
                    self.storage = 0
                else:
                    support_ship_1.storage = support_ship_1.max_storage
                    self.storage = self.storage - support_ship_1.max_storage

                self.call_num = self.call_num + 1

        elif (support_ship_2.arrived_supplybase == 1) or (
            self.call_ship2 == 1
        ):  # support_ship_1がダメでsupport_ship_2が活動可能な場合
            self.brance_condition = "call ship2"
            self.call_ship2 = 1
            support_ship_2.get_next_ship_state(
                self.locate, year, current_time, time_step
            )

            if support_ship_2.arrived_storagebase == 1:
                self.call_ship2 = 0
                if self.storage <= support_ship_2.max_storage:
                    support_ship_2.storage = support_ship_2.storage + self.storage
                    self.storage = 0
                else:
                    support_ship_2.storage = support_ship_2.max_storage
                    self.storage = self.storage - support_ship_2.max_storage

                self.call_num = self.call_num + 1
        else:  # 両方ダメな場合
            self.brance_condition = "can't call anyone"

    def operation_base(
        self, TPGship1, support_ship_1, support_ship_2, year, current_time, time_step
    ):
        """
        ############################ def operation_base ############################

        [ 説明 ]

        中継貯蔵拠点の運用を行う関数。

        """

        # 貯蔵量の更新
        self.storage_elect(TPGship1)
        self.brance_condition = "while in storage"

        # supportSHIPの寄港動作完遂までは動かす。呼び出しもキャンセル。
        if support_ship_1.arrived_supplybase == 0:
            support_ship_1.get_next_ship_state(
                self.locate, year, current_time, time_step
            )

        if support_ship_2.arrived_supplybase == 0:
            support_ship_2.get_next_ship_state(
                self.locate, year, current_time, time_step
            )

        judge = support_ship_1.max_storage * (self.call_per / 100)
        if self.storage >= judge:
            self.supply_elect(
                support_ship_1, support_ship_2, year, current_time, time_step
            )
