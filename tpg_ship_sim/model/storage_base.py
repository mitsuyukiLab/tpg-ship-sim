class storage_BASE:
    """
    ############################### class storage_BASE ###############################

    [ 説明 ]

    このクラスは中継貯蔵拠点を作成するクラスです。

    主にTPGshipが生成した電力や水素を貯蔵し、補助船に渡します。

    中継貯蔵拠点の能力や状態量もここで定義されることになります。

    ##############################################################################

    引数 :
        year (int) : シミュレーションを行う年
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)
        supportSHIP1 (class) : support_SHIPクラスのインスタンスその1
        supportSHIP2 (class) : support_SHIPクラスのインスタンスその2
        TPGship1 (class) : TPGshipクラスのインスタンスその1

    属性 :
        max_storage (float) : 中継貯蔵拠点の蓄電容量の上限値
        storage (float) : 中継貯蔵拠点のその時刻での蓄電量
        call_num (int) : supportSHIPを読んだ回数
        call_ship1 (int) : supportSHIP1を呼ぶフラグ
        call_ship2 (int) : supportSHIP1を呼ぶフラグ
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

    # 南鳥島
    lat = 24
    lon = 153
    locate = (lat, lon)

    ####################################  メソッド  ######################################

    def storage_elect(self, TPGship1):
        """
        ############################ def storage_elect ############################

        [ 説明 ]

        中継貯蔵拠点がTPGshipから発電成果物を受け取り、蓄電容量を更新する関数です。

        TPGshipが拠点に帰港した時にのみ増加する。それ以外は出力の都合で、常に0を受け取っている。

        """

        self.storage = self.storage + TPGship1.supply_elect

        if TPGship1.supply_elect > 0:
            TPGship1.supply_elect = 0

    def supply_elect(self, supportSHIP1, supportSHIP2, year, current_time, time_step):
        """
        ############################ def supply_elect ############################

        [ 説明 ]

        中継貯蔵拠点がsupportSHIPを呼び出し、貯蔵しているエネルギーを渡す関数。

        呼ぶまでの関数なので、読んだ後のsupportSHIPが帰るフェーズは別で記載している。

        """
        if (supportSHIP1.arrived_supplybase == 1) or (
            self.call_ship1 == 1
        ):  # supportSHIP1が活動可能な場合
            self.brance_condition = "call ship1"
            self.call_ship1 = 1
            supportSHIP1.get_next_ship_state(self.locate, year, current_time, time_step)

            if supportSHIP1.arrived_storagebase == 1:
                self.call_ship1 = 0
                if self.storage <= supportSHIP1.max_storage:
                    supportSHIP1.storage = supportSHIP1.storage + self.storage
                    self.storage = 0
                else:
                    supportSHIP1.storage = supportSHIP1.max_storage
                    self.storage = self.storage - supportSHIP1.max_storage

                self.call_num = self.call_num + 1

        elif (supportSHIP2.arrived_supplybase == 1) or (
            self.call_ship2 == 1
        ):  # supportSHIP1がダメでsupportSHIP2が活動可能な場合
            self.brance_condition = "call ship2"
            self.call_ship2 = 1
            supportSHIP2.get_next_ship_state(self.locate, year, current_time, time_step)

            if supportSHIP2.arrived_storagebase == 1:
                self.call_ship2 = 0
                if self.storage <= supportSHIP2.max_storage:
                    supportSHIP2.storage = supportSHIP2.storage + self.storage
                    self.storage = 0
                else:
                    supportSHIP2.storage = supportSHIP2.max_storage
                    self.storage = self.storage - supportSHIP2.max_storage

                self.call_num = self.call_num + 1
        else:  # 両方ダメな場合
            self.brance_condition = "can't call anyone"

    def operation_base(
        self, TPGship1, supportSHIP1, supportSHIP2, year, current_time, time_step
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
        if supportSHIP1.arrived_supplybase == 0:
            supportSHIP1.get_next_ship_state(self.locate, year, current_time, time_step)

        if supportSHIP2.arrived_supplybase == 0:
            supportSHIP2.get_next_ship_state(self.locate, year, current_time, time_step)

        judge = supportSHIP1.max_storage * (self.call_per / 100)
        if self.storage >= judge:
            self.supply_elect(supportSHIP1, supportSHIP2, year, current_time, time_step)
