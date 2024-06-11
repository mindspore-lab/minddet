"""Database sampler builder"""
import pickle

from src.builder import preprocess_builder
from src.core.preprocess import DataBasePreprocessor
from src.core.sample_ops import DataBaseSampler


def build(sampler_config):
    """build sampler"""
    cfg = sampler_config
    groups = cfg.get("sample_groups")
    if groups is None:
        raise KeyError('"sample_groups" must be defined if "database_sampler" is used')
    preproc = [
        preprocess_builder.build_db_preprocess({k: v})
        for k, v in cfg["database_prep_steps"].items()
    ]
    db_prepor = DataBasePreprocessor(preproc)
    rate = cfg["rate"]
    grot_range = cfg["global_random_rotation_range_per_object"]
    groups = [{groups["name_to_max_num"]["key"]: groups["name_to_max_num"]["value"]}]
    info_path = cfg["database_info_path"]
    with open(info_path, "rb") as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if not grot_range:
        grot_range = None
    sampler = DataBaseSampler(db_infos, groups, db_prepor, rate, grot_range)
    return sampler
