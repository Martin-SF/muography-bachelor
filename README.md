# muography-bachelor

Here are instructions to reproduce my bachelor thesis simulation

## preparations

### clone repo

`git clone https://github.com/Martin-SF/muography-bachelor`

### Set up the conda environment
`conda create --name m1 python=3.10 pip cmake`

`conda activate m1`

`pip install -r muography-bachelor/requirements.txt`

### environment Variables (configure the paths for YOUR system)

(This is necessary so that the compiled Ecomug module and config.py can be found)

`conda env config vars set PYTHONPATH="$HOME/muography-bachelor/computation/EcoMug_pybind11/build:$PYTHONPATH"`

`conda env config vars set PYTHONPATH="$HOME/muography-bachelor/computation:$PYTHONPATH"`

(This can be advantageous if `libpython` is not found (for example PROPOSAL)

`conda env config vars set LD_LIBRARY_PATH="$HOME/.local/anaconda3/envs/m1/lib:$LD_LIBRARY_PATH"`

### Build EcoMug

`cd muography-bachelor/computation/EcoMug_pybind11`

`mkdir build`

`cd build`

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make`


### Ready

Don't forget to restart the conda environment! (also VSCode as whole is a good idea)

Now you should be able to run `computation/computation.ipynb`!

# Continuation Project


[RUMS - RUhrgebiet Myographie Simulation](https://github.com/Martin-SF/RUMS)

Currently in active developement (10.2023)

# Corrections

Ecomug expects elevation angles, not azimuth angles, so the results in this thesis are incorrect with respect to the azimuth distribution produced by EcoMug. This results in the overall estimate of muon rates being incorrect as well. 

The relative rate differences are still vaild.