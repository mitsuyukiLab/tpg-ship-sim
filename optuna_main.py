import csv

import hydra
import optuna
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from tpg_ship_sim import optuna_simulator, utils
from tpg_ship_sim.model import forecaster, storage_base, support_ship, tpg_ship


# 進捗バーを更新するコールバック関数を定義
class TqdmCallback(object):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Optuna Trials")

    def __call__(self, study, trial):
        self.pbar.update(1)


def run_simulation(cfg):
    typhoon_data_path = cfg.env.typhoon_data_path
    simulation_start_time = cfg.env.simulation_start_time
    simulation_end_time = cfg.env.simulation_end_time

    output_folder_path = HydraConfig.get().run.dir

    tpg_ship_param_log_file_name = cfg.output_env.tpg_ship_param_log_file_name
    temp_tpg_ship_param_log_file_name = (
        "temp_" + cfg.output_env.tpg_ship_param_log_file_name
    )

    # TPG ship
    initial_position = cfg.tpg_ship.initial_position
    hull_num = cfg.tpg_ship.hull_num
    storage_method = cfg.tpg_ship.storage_method
    max_storage_wh = cfg.tpg_ship.max_storage_wh
    electric_propulsion_max_storage_wh = cfg.tpg_ship.electric_propulsion_max_storage_wh
    elect_trust_efficiency = cfg.tpg_ship.elect_trust_efficiency
    MCH_to_elect_efficiency = cfg.tpg_ship.MCH_to_elect_efficiency
    elect_to_MCH_efficiency = cfg.tpg_ship.elect_to_MCH_efficiency
    generator_output_w = cfg.tpg_ship.generator_output_w
    generator_efficiency = cfg.tpg_ship.generator_efficiency
    generator_drag_coefficient = cfg.tpg_ship.generator_drag_coefficient
    generator_pillar_chord = cfg.tpg_ship.generator_pillar_chord
    generator_pillar_max_tickness = cfg.tpg_ship.generator_pillar_max_tickness
    generator_pillar_width = cfg.tpg_ship.generator_pillar_width
    generator_num = cfg.tpg_ship.generator_num
    sail_num = cfg.tpg_ship.sail_num
    sail_area = cfg.tpg_ship.sail_area
    sail_steps = cfg.tpg_ship.sail_steps
    sail_weight = cfg.tpg_ship.sail_weight
    ship_return_speed_kt = cfg.tpg_ship.ship_return_speed_kt
    ship_max_speed_kt = cfg.tpg_ship.ship_max_speed_kt
    ship_generate_speed_kt = cfg.tpg_ship.ship_generate_speed_kt
    forecast_weight = cfg.tpg_ship.forecast_weight
    typhoon_effective_range = cfg.tpg_ship.typhoon_effective_range
    govia_base_judge_energy_storage_per = (
        cfg.tpg_ship.govia_base_judge_energy_storage_per
    )
    judge_time_times = cfg.tpg_ship.judge_time_times

    tpg_ship_1 = tpg_ship.TPG_ship(
        initial_position,
        hull_num,
        storage_method,
        max_storage_wh,
        electric_propulsion_max_storage_wh,
        elect_trust_efficiency,
        MCH_to_elect_efficiency,
        elect_to_MCH_efficiency,
        generator_output_w,
        generator_efficiency,
        generator_drag_coefficient,
        generator_pillar_chord,
        generator_pillar_max_tickness,
        generator_pillar_width,
        generator_num,
        sail_num,
        sail_area,
        sail_steps,
        sail_weight,
        ship_return_speed_kt,
        ship_max_speed_kt,
        ship_generate_speed_kt,
        forecast_weight,
        typhoon_effective_range,
        govia_base_judge_energy_storage_per,
        judge_time_times,
    )

    # Forecaster
    forecast_time = cfg.forecaster.forecast_time
    forecast_error_slope = cfg.forecaster.forecast_error_slope
    typhoon_path_forecaster = forecaster.Forecaster(forecast_time, forecast_error_slope)

    # Storage base
    storage_base_locate = cfg.storage_base.locate
    storage_base_max_storage_wh = cfg.storage_base.max_storage_wh
    st_base = storage_base.Storage_base(
        storage_base_locate, storage_base_max_storage_wh
    )

    # Support ship 1
    support_ship_1_supply_base_locate = cfg.support_ship_1.supply_base_locate
    support_ship_1_max_storage_wh = cfg.support_ship_1.max_storage_wh
    support_ship_1_max_speed_kt = cfg.support_ship_1.ship_speed_kt
    support_ship_1 = support_ship.Support_ship(
        support_ship_1_supply_base_locate,
        support_ship_1_max_storage_wh,
        support_ship_1_max_speed_kt,
    )

    # Support ship 2
    support_ship_2_supply_base_locate = cfg.support_ship_2.supply_base_locate
    support_ship_2_max_storage_wh = cfg.support_ship_2.max_storage_wh
    support_ship_2_max_speed_kt = cfg.support_ship_2.ship_speed_kt
    support_ship_2 = support_ship.Support_ship(
        support_ship_2_supply_base_locate,
        support_ship_2_max_storage_wh,
        support_ship_2_max_speed_kt,
    )

    # Run simulation
    optuna_simulator.simulate(
        simulation_start_time,
        simulation_end_time,
        tpg_ship_1,
        typhoon_path_forecaster,
        st_base,
        support_ship_1,
        support_ship_2,
        typhoon_data_path,
        output_folder_path,
        tpg_ship_param_log_file_name,
        temp_tpg_ship_param_log_file_name,
    )

    print(tpg_ship_1.total_gene_elect)

    return tpg_ship_1.total_gene_elect


# 探索範囲の指定用関数
def objective(trial):
    config = hydra.compose(config_name="config")

    # config.tpg_ship.hull_num = trial.suggest_int("hull_num", 1, 2)
    # config.tpg_ship.storage_method = trial.suggest_int("storage_method", 1, 2)
    config.tpg_ship.max_storage_wh = trial.suggest_int(
        "max_storage_wh", 50000000000, 300000000000
    )
    config.tpg_ship.electric_propulsion_max_storage_wh = trial.suggest_int(
        "electric_propulsion_max_storage_wh", 20000000000, 60000000000
    )
    # config.tpg_ship.elect_trust_efficiency = trial.suggest_float("elect_trust_efficiency", 0.7, 0.9)
    # config.tpg_ship.MCH_to_elect_efficiency = trial.suggest_float("MCH_to_elect_efficiency", 0.4, 0.6)
    # config.tpg_ship.elect_to_MCH_efficiency = trial.suggest_float("elect_to_MCH_efficiency", 0.7, 0.9)
    config.tpg_ship.sail_num = trial.suggest_int("sail_num", 10, 60)
    # config.tpg_ship.sail_area = trial.suggest_int("sail_area", 700, 1000)
    config.tpg_ship.sail_steps = trial.suggest_int("sail_steps", 3, 7)
    config.tpg_ship.ship_return_speed_kt = trial.suggest_int(
        "ship_return_speed_kt", 4, 15
    )
    config.tpg_ship.forecast_weight = trial.suggest_int("forecast_weight", 10, 90)
    # config.tpg_ship.typhoon_effective_range = trial.suggest_int("typhoon_effective_range", 50, 150)
    config.tpg_ship.govia_base_judge_energy_storage_per = trial.suggest_int(
        "govia_base_judge_energy_storage_per", 10, 90
    )
    config.tpg_ship.judge_time_times = trial.suggest_float("judge_time_times", 1.0, 1.5)

    # シミュレーションを実行
    total_generation = run_simulation(config)

    return total_generation


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    # 結果保存用のCSVファイルを初期化
    output_folder_path = HydraConfig.get().run.dir
    tpg_ship_param_log_file_name = cfg.output_env.tpg_ship_param_log_file_name

    study = optuna.create_study(direction="maximize")
    # 結果保存用のCSVファイルを初期化
    final_csv = output_folder_path + "/" + tpg_ship_param_log_file_name

    columns = [
        ("ship_lat", pl.Float64),
        ("ship_lon", pl.Float64),
        ("hull_num", pl.Int64),
        ("storage_method", pl.Int64),
        ("max_storage", pl.Float64),
        ("electric_propulsion_max_storage_wh", pl.Float64),
        ("elect_trust_efficiency", pl.Float64),
        ("MCH_to_elect_efficiency", pl.Float64),
        ("elect_to_MCH_efficiency", pl.Float64),
        ("generator_output_w", pl.Float64),
        ("generator_efficiency", pl.Float64),
        ("generator_drag_coefficient", pl.Float64),
        ("generator_pillar_chord", pl.Float64),
        ("generator_pillar_max_tickness", pl.Float64),
        ("generator_pillar_width", pl.Float64),
        ("generator_num", pl.Int64),
        ("sail_num", pl.Int64),
        ("sail_area", pl.Float64),
        ("sail_steps", pl.Int64),
        ("sail_weight", pl.Float64),
        ("nomal_ave_speed", pl.Float64),
        ("max_speed", pl.Float64),
        ("generating_speed_kt", pl.Float64),
        ("forecast_weight", pl.Float64),
        ("typhoon_effective_range", pl.Float64),
        ("govia_base_judge_energy_storage_per", pl.Float64),
        ("judge_time_times", pl.Float64),
        ("total_gene_elect", pl.Float64),
    ]

    # Create an empty DataFrame with the specified schema
    df = pl.DataFrame(schema=columns)

    df.write_csv(final_csv)

    # 進捗バーのコールバックを使用してoptimizeを実行
    study.optimize(objective, n_trials=100, callbacks=[TqdmCallback(total=100)])

    # 最良の試行を出力
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
