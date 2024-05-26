from tpg_ship_sim import (
    simulator,
    utils,
)
import polars as pl

from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


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

    simulator.simulate(
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
