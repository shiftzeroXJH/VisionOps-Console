# Model weights

Place official YOLOv8, YOLO11 and YOLO26 `.pt` files in this directory.
Missing official weights are downloaded here when training starts. Semantic
segmentation uses YOLO26 `yolo26{n,s,m,l,x}-sem.pt` weights.

Weights, download staging files and locks are not committed. Custom weights can
still be selected in training parameters. Remote hosts cache official weights
under their configured working directory's `models` folder.
