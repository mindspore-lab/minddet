"""similarity calculator builder"""
from src.core import region_similarity


def build(similarity_config):
    """build similarity config"""
    similarity_type = similarity_config
    if similarity_type == "rotate_iou_similarity":
        return region_similarity.RotateIouSimilarity()
    if similarity_type == "nearest_iou_similarity":
        return region_similarity.NearestIouSimilarity()
    if similarity_type == "distance_similarity":
        cfg = similarity_config.distance_similarity
        return region_similarity.DistanceSimilarity(
            distance_norm=cfg.distance_norm,
            with_rotation=cfg.with_rotation,
            rotation_alpha=cfg.rotation_alpha,
        )
    raise ValueError("unknown similarity type")
