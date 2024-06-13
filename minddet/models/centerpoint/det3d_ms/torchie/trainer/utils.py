import sys

from det3d_ms import torchie


def obj_from_dict(info, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type

    Args:
        info (dict): Object types and arguments
        parent (:class:`modules`):
        default_args (dict, optional):
    """
    assert isinstance(info, dict) and "type" in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop("type")
    if torchie.is_str(obj_type):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)
