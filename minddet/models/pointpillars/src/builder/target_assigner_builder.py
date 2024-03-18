"""target assigner builder"""
from src.builder import anchor_generator_builder, similarity_calculator_builder
from src.core.target_assigner import TargetAssigner


def build(target_assigner_config, box_coder, point_cloud_range, use_self_train=True):
    """build target assigner"""
    anchor_cfg = target_assigner_config["anchor_generators"]
    anchor_generators = []
    for cl in anchor_cfg:
        for a_type, a_cfg in anchor_cfg[cl].items():
            anchor_generator = anchor_generator_builder.build(
                a_cfg, a_type, point_cloud_range, use_self_train
            )
            anchor_generators.append(anchor_generator)
    similarity_calc = similarity_calculator_builder.build(
        target_assigner_config["region_similarity_calculator"]
    )
    positive_fraction = target_assigner_config["sample_positive_fraction"]
    if positive_fraction < 0:
        positive_fraction = None
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        region_similarity_calculator=similarity_calc,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config["sample_size"],
    )
    return target_assigner
