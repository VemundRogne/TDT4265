conda create -n tdt4265 python=3.8 -y
conda activate tdt4265
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda clean -t -y
conda install tqdm -y
conda install -c conda-forge scikit-image -y
conda install -c conda-forge pybind11 -y
conda clean -t -y
pip intall -r requirements.txt --yes
pip install click -y
pip install tensorboard -y