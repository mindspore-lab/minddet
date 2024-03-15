"""preproc builder"""
import src.core.preprocess as prep


def build_db_preprocess(db_prep_config):
    """build db preprocess"""
    prep_type = list(db_prep_config.keys())[0]

    if prep_type == "filter_by_difficulty":
        cfg = db_prep_config["filter_by_difficulty"]
        return prep.DBFilterByDifficulty(list(cfg["removed_difficulties"]))
    if prep_type == "filter_by_min_num_points":
        cfg = db_prep_config["filter_by_min_num_points"]
        return prep.DBFilterByMinNumPoint(
            {cfg["min_num_point_pairs"]["key"]: cfg["min_num_point_pairs"]["value"]}
        )
    raise ValueError("unknown database prep type")
