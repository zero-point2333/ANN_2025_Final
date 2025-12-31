# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        else:
            break
    if hasattr(dataset, "coco"):
        return dataset.coco


def build_dataset(image_set, args, datasetinfo):
    if datasetinfo["dataset_mode"] == 'coco':
        return build_coco(image_set, args, datasetinfo)
    if datasetinfo["dataset_mode"] == 'odvg':
        from .odvg import build_odvg
        return build_odvg(image_set, args, datasetinfo)
    raise ValueError(f'dataset {args.dataset_file} not supported')
