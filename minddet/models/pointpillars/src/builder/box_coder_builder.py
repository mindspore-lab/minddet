"""Box coder builder"""
from src.core.box_coders import BevBoxCoder, GroundBox3dCoder


def build(box_coder_config):
    """build box coder"""
    box_coder_type = box_coder_config["type"]
    if box_coder_type == "ground_box3d_coder":
        return GroundBox3dCoder(
            box_coder_config["linear_dim"], box_coder_config["encode_angle_vector"]
        )
    if box_coder_type == "bev_box_coder":
        return BevBoxCoder(
            box_coder_config["linear_dim"],
            box_coder_config["encode_angle_vector"],
            box_coder_config["z_fixed"],
            box_coder_config["h_fixed"],
        )
    raise ValueError("unknown box_coder type")
