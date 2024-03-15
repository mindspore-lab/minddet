"""Model builder"""
from src.pointpillars import PointPillarsNet


def build(model_cfg, voxel_generator, target_assigner, use_self_train=True):
    """build model"""
    vfe_num_filters = model_cfg["voxel_feature_extractor"]["num_filters"]
    grid_size = voxel_generator.grid_size
    output_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    voxel_size = list(map(float, voxel_generator.voxel_size))
    pc_range = list(map(float, voxel_generator.point_cloud_range))
    pointpillarsnet = PointPillarsNet(
        target_assigner.box_coder,
        output_shape=output_shape,
        use_self_train=use_self_train,
        num_class=model_cfg["num_class"],
        num_input_features=model_cfg["num_point_features"],
        vfe_num_filters=model_cfg["voxel_feature_extractor"]["num_filters"],
        with_distance=model_cfg["voxel_feature_extractor"]["with_distance"],
        rpn_layer_nums=model_cfg["rpn"]["layer_nums"],
        rpn_layer_strides=model_cfg["rpn"]["layer_strides"],
        rpn_num_filters=model_cfg["rpn"]["num_filters"],
        rpn_upsample_strides=model_cfg["rpn"]["upsample_strides"],
        rpn_num_upsample_filters=model_cfg["rpn"]["num_upsample_filters"],
        use_norm=model_cfg["use_norm"],
        use_direction_classifier=model_cfg["use_direction_classifier"],
        encode_background_as_zeros=model_cfg["encode_background_as_zeros"],
        num_anchor_per_loc=target_assigner.num_anchors_per_location,
        code_size=target_assigner.box_coder.code_size,
        use_bev=model_cfg["use_bev"],
        voxel_size=voxel_size,
        pc_range=pc_range,
        use_sigmoid_score=model_cfg["use_sigmoid_score"],
        pre_max_size=model_cfg["nms_pre_max_size"],
    )

    return pointpillarsnet
