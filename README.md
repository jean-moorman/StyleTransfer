# Perceptual Loss Style Transfer 
(paper can be found [here](https://arxiv.org/abs/1603.08155))
Step 1:

Install Anaconda3 here: https://www.anaconda.com/products/individual \
Miniconda3 (much lighter): https://docs.conda.io/en/latest/miniconda.html

Step 2:

conda create -n pytorch \
conda activate pytorch \
conda install pytorch torchvision cudatoolkit=YOUR_CUDA_VERSION -c pytorch

P.S. To find your CUDA version, run nvidia-smi in terminal

Training CMD Example (You should play around with content-weight to style-weight ratios), takes around 4 hrs on GPU:

python neural_style/fast-style-transfer.py train --cuda 1 --dataset /etc/detectron2/coco --style-image images/style-images/starrySky.jpeg --save-model-dir neural_style/saves --content-weight 1e2 --style-weight 5e7

Evaluation CMD Example:

python neural_style/fast-style-transfer.py eval --cuda 1 --model neural_style/saves/starrySky.model --content-image images/content-images/GroupPic.jpg --output-image images/output-images/SpaceGroup.jpg


P.S.S. I included some pretrained style models with varying content-style ratios in the saves folder
