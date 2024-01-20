# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/levy-tech-spark/AViD
# Copyright (c) 2024 Levy Hu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .GroundingDINO import build_groundingdino


def build_model(args):
    # we use register to maintain models from catdet6 on.
    from .registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model
