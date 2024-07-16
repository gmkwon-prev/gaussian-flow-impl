conda create -n gaussian_flow python=3.9
source activate gaussian_flow
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install plyfile tqdm tensorboard
pip install scipy
pip install submodules/diff-gaussian-rasterization/   
pip install submodules/simple-knn 