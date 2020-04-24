# PerceptualLoss
Implementation of fast style transfer "Perceptual Loss" neural architecture w/ instance normalization

Training CMD Example (You should play around with content-weight to style-weight ratios), takes around 4 hrs on GPU:

python neural_style/fast-style-transfer.py train --cuda 1 --dataset /etc/detectron2/coco --style-image images/style-images/starrySky.jpeg --save-model-dir neural_style/saves --content-weight 1e2 --style-weight 5e7

Evaluation CMD Example:

python neural_style/fast-style-transfer.py eval --cuda 1 --model neural_style/saves/starrySky.model --content-image images/content-images/GroupPic.jpg --output-image images/output-images/SpaceGroup.jpg
