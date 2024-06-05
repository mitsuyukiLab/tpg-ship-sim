import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from tpg_ship_sim import simulator, utils
from tpg_ship_sim.model import forecaster, storage_base, support_ship, tpg_ship


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:

    typhoon_data_path = cfg.env.typhoon_data_path

    output_folder_path = HydraConfig.get().run.dir

    tpg_ship_log_file_name = cfg.output_env.tpg_ship_log_file_name
    storage_base_log_file_name = cfg.output_env.storage_base_log_file_name
    support_ship_1_log_file_name = cfg.output_env.support_ship_1_log_file_name
    support_ship_2_log_file_name = cfg.output_env.support_ship_2_log_file_name
    png_map_folder_name = cfg.output_env.png_map_folder_name
    png_graph_folder_name = cfg.output_env.png_graph_folder_name
    png_map_graph_folder_name = cfg.output_env.png_map_graph_folder_name

    progress_bar = tqdm(total=6, desc=output_folder_path)

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
    base_locate = cfg.storage_base.locate
    st_base_max_storage_wh = cfg.storage_base.max_storage_wh
    st_base = storage_base.Storage_base(base_locate, st_base_max_storage_wh)

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

    simulator.simulate(
        tpg_ship_1,  # TPG ship
        typhoon_path_forecaster,  # Forecaster
        st_base,  # Storage base
        support_ship_1,  # Support ship 1
        support_ship_2,  # Support ship 2
        typhoon_data_path,
        output_folder_path + "/" + tpg_ship_log_file_name,
        output_folder_path + "/" + storage_base_log_file_name,
        output_folder_path + "/" + support_ship_1_log_file_name,
        output_folder_path + "/" + support_ship_2_log_file_name,
    )
    progress_bar.update(1)

    utils.draw_map(
        typhoon_data_path,
        output_folder_path + "/" + tpg_ship_log_file_name,
        output_folder_path + "/" + storage_base_log_file_name,
        output_folder_path + "/" + support_ship_1_log_file_name,
        output_folder_path + "/" + support_ship_2_log_file_name,
        output_folder_path + "/" + png_map_folder_name,
    )
    progress_bar.update(1)

    utils.draw_graph(
        typhoon_data_path,
        output_folder_path + "/" + tpg_ship_log_file_name,
        output_folder_path + "/" + storage_base_log_file_name,
        output_folder_path + "/" + png_graph_folder_name,
    )
    progress_bar.update(1)

    # TODO : Just for getting the length of simulation data.
    sim_data_length = len(
        pl.read_csv(output_folder_path + "/" + tpg_ship_log_file_name)
    )

    utils.merge_map_graph(
        sim_data_length,
        output_folder_path + "/" + png_map_folder_name,
        output_folder_path + "/" + png_graph_folder_name,
        output_folder_path + "/" + png_map_graph_folder_name,
    )
    progress_bar.update(1)

    # create_movie
    utils.create_movie(
        output_folder_path + "/" + png_map_graph_folder_name,
        output_folder_path,
    )
    progress_bar.update(1)

    # finish
    progress_bar.update(1)


if __name__ == "__main__":
    main()
