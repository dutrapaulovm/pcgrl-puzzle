python -m pip uninstall pcgrl

python -m pip install --upgrade pip setuptools wheel

pip install gym
pip install stable-baselines3==1.6.2
pip install sb3-contrib==1.6.2
pip install pygame==2.0.0 
pip install ipython
pip install opencv-python
pip install imageio
pip install pandas

#python -m pip install dist/pcgrl-0.0.70.10-py3-none-any.whl

pip install -r requirements.txt

python setup.py bdist_wheel