# muography-bachelor

Here are instructions to reproduce my bachelors thesis simulation

## preparations

### clone repo

`git clone https://github.com/Martin-SF/muography-bachelor`

### set up conda environment
`conda create -n m1 python=3.10 pip`

`conda activate m1`

`cd muography-bachelor`

`pip install -r requirements.txt`


### compile EcoMug
`cd muography-bachelor/computation/EcoMug_pybind11`

`mkdir build`

`cd build`

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make`

### variables (configure for your system)
`conda env config vars set PYTHONPATH="$HOME/muography-bachelor/computation/EcoMug_pybind11/build:$PYTHONPATH"`

`conda env config vars set LD_LIBRARY_PATH="$HOME/.local/anaconda3/envs/m1/lib:$LD_LIBRARY_PATH"`

## ready

now you can start the `computation/computation.ipynb` and continue there 
conda create -n m1 python=3.10 pip --y; conda activate m1; pip install -r requirements.txt --y