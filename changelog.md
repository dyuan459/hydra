# Changelog
1. Initial configurations
    1. get rid of scipy from requirements.txt

2. YOLOv3 integration (commit f54b69b)
    1. Added `YOLOv3.py`
            - Taken from Yolov3 project
            - Integrates `SubnetConv` support so the model can be used with HYDRA's
                pruning subnet layers.
    2. Added `yolov3.cfg`
            - From yolov3 project
    3. Updated `train.py`
            - Imports and uses `load_model` to instantiate the YOLOv3 model.
            - Replaces the previous classification model construction with YOLOv3
                loading; adds extensive developer notes about detection-specific
                concerns (COCO dataloader compatibility, loss vs. classification,
                checkpoint key mapping, dense->subnet conversion, initialization of
                popup scores, and save/load semantics for pruned weights).
            - Adds guidance to ensure `prepare_model` does not inadvertently
                treat detection-model buffers (grids, anchors) as trainable params.
    4. Updated `requirements.txt`
            - Added `pytorch` entry.
    5. Compatibility notes
            - Loading PyTorch checkpoint formats now handles both plain state_dict
                files and checkpoint dicts (uses `strict=False` where appropriate).
            - Emphasizes use of `model.state_dict()` when capturing the loaded
                model state for subsequent operations (avoids assuming a 'state_dict'
                key exists in the checkpoint container).
            - SNIP/score initialization, pruning and detection-head handling require
                special attention; comments were added in `train.py` to guide future
                edits.

3. Misc
    1. Minor inline comments and TODOs added to help integrate YOLOv3 with the
            HYDRA pruning workflow and with COCO-style datasets.
    3. augmentations.py
    4. all utils except already imported
    5. Currently testing with 20 train and validation weight for convinience. Still running into various issues from missed compatibility issues.

