# muography-bachelor

Here are instructions to reproduce my bachelor thesis simulation

## preparations

### clone repo

`git clone https://github.com/Martin-SF/muography-bachelor`

### Set up the conda environment
`conda create --name m1 python=3.10 pip`

`conda activate m1`

`cd muography-bachelor`

`pip install -r requirements.txt`


### Build EcoMug
`cd muography-bachelor/computation/EcoMug_pybind11`

`mkdir build`

`cd build`

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make`

### Variables (configure for YOUR system)
`conda env config vars set PYTHONPATH="$HOME/muography-bachelor/computation/EcoMug_pybind11/build:$PYTHONPATH"`

`conda env config vars set PYTHONPATH="$HOME/muography-bachelor/computation:$PYTHONPATH"`

(probably not needed!) `conda env config vars set LD_LIBRARY_PATH="$HOME/.local/anaconda3/envs/m1/lib:$LD_LIBRARY_PATH"`

## Done

Don't forget to restart the conda environment!

Now you should be able to run the `computation/computation.ipynb`!
