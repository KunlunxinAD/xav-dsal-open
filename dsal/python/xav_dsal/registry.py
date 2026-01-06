_OPS = {}

def register_op(name: str, func, for_mmcv: bool = False):
    """注册算子"""
    if name in _OPS:
        return  # 避免重复注册
    _OPS[name] = {"func": func, "for_mmcv": for_mmcv}


def get_all_ops():
    return _OPS


def get_mmcv_patch_ops():
    return [v["func"] for v in _OPS.values() if v["for_mmcv"]]