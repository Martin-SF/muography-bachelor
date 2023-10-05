# muography-bachelor

Here are instructions to reproduce my bachelor thesis simulation

## preparations

### clone repo

`git clone https://github.com/Martin-SF/muography-bachelor`

### Set up the conda environment
`conda create --name m1 python=3.10 pip`

`conda activate m1`

`pip install -r muography-bachelor/requirements.txt`

### environment Variables (configure the paths for YOUR system)

(This is necessary so that the compiled Ecomug module can be found)

`conda env config vars set PYTHONPATH="$HOME/muography-bachelor/computation/EcoMug_pybind11/build:$PYTHONPATH"`

(This can be advantageous if Libpython is not found when a PROPOSAL is imported or other programs fail weirdly (cmake) 

`conda env config vars set LD_LIBRARY_PATH="$HOME/.local/anaconda3/envs/m1/lib:$LD_LIBRARY_PATH"`

### Build EcoMug
`cd muography-bachelor/computation/EcoMug_pybind11`

`mkdir build`

`cd build`

if this makes problems, maybe look into the env_3.10 environment requirement.txt. This environment is the only one that will compile Ecomug when using `phobos`...

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make`


## Done

Don't forget to restart the conda environment! (Als VSCode as whole if you use it (trust me...))

Now you should be able to run `computation/computation.ipynb`!
