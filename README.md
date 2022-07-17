# muography-bachelor


# preparations
## set up environment
`conda create -n m1`

`conda activate m1`

`conda install python=3.10` 

`pip install numpy pandas matplotlib numba distributed tqdm scipy uncertainties prettytable proposal tables bokeh jupyter-server-proxy`

### variables (configure for your system)
`conda env config vars set LD_LIBRARY_PATH="$HOME/.local/anaconda3/envs/m1/lib:$LD_LIBRARY_PATH"`

`conda env config vars set PYTHONPATH="$HOME/test/muography-bachelor/computation/EcoMug_pybind11/build:$PYTHONPATH"`

### clone repo

`git clone https://github.com/Martin-SF/muography-bachelor`

### compile EcoMug
`cd muography-bachelor/computation/EcoMug_pybind11`

`mkdir build`

`cd build`

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make`

## ready

now you can start the `computation/computation.ipynb` and follow the steps there 
