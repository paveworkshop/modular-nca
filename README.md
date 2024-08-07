# modular-nca
**An experimental Python-based framework for training modular neural cellular automata.**

 ![Neural network visualisation](https://github.com/paveworkshop/modular-nca/blob/main/previews/neural-network.png)  
_The neural network embedded inside each simulation 'pixel', which mimics a gene regulatory network._
 
**A petri dish for self-taught growth dynamics**  
The cellular automata 'substrate' is composed of 3 visible channels (rgb), 1 alpha channel and 12 hidden channels (h0-h12) which the neural network within each cell learns to incrementally update towards a stable pattern.  

![Neural network visualisation](https://github.com/paveworkshop/modular-nca/blob/main/previews/hex-leaf-mono-7-0-64-16-9241-3826-1721073373.gif)  

_In this discarded experiment, the NCA learnt to 'cheat' the task of growing a leaf by using the walls of the substrate._

## Contents
- [Results](#Results)
- [Usage](#Usage)
- [Credits](#Credits)
- [License](#License)

## Results
A modular NCA trained on a simplified leaf image.  

**Training dataset**  
 ![Leaf training set](https://github.com/paveworkshop/modular-nca/blob/main/training_datasets/leaf-mono-thumbnail.png)
 
**Training result**  
After ~4500 epochs of training, the model is persisently able to grow a persistent leaf pattern in 500 steps, albeit not in a way that is biologically accurate.  
![Leaf training result](https://github.com/paveworkshop/modular-nca/blob/main/previews/leaf-mono-small-training-result-visible.gif)

**Training Progression**  
The model slowly learns to use all 16 channels to encode helpful cues (like edges) that help cells express the right colour in the right places. First it learns to grow the solid shape, followed by finer details of the venation pattern.  
![Leaf training progression](https://github.com/paveworkshop/modular-nca/blob/main/previews/leaf-mono-small-training-progression-all.gif)

## Usage
- [eval_nca.py](eval_nca.ipynb)
- [nca_app.py](nca_app.py)

In order to preview the results of the 'leaf-mono-small' experiment, without training the models again yourself (very slow), please use the 'eval_nca.ipynb' notebook, by cloning this repository, in order to maintain the dependencies and file structure.
The [modular_nca](modular_nca) folder contains the Python source code.

The GUI desktop application 'nca_app.py' is used to train models and recording previews. It is not expected you will train models, but for completeness, it can be activated via the command line using _python3 nca_app.py <app_mode>_, where available app_modes are:

**App Modes**  
-1 = train model with current configuration in nca_app.py and config.py.  
0 = preview images pasted on the grid cells, like the training target image.  
1+ = evaluate model (run simulation) using latest training checkpoint, previewing selecting grid channels as below.  

**Channel Preview Modes** (for the current model, with 16 channels total)   
1 = rgb   
2 = alpha  
3 = hidden channels (h0-3)    
4 = hidden channels (h3-6)  
5 = hidden channels (h6-9)  
6 = hidden channels (h9-12)  
7 = all channels  

## Credits
The NCA architecture was based on [this](https://distill.pub/2020/growing-ca/) paper:  
Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020). Growing neural cellular automata. Distill, 5(2), e23. 

With helpful insights from this paper too:  
Catrina, S., Catrina, M., Băicoianu, A., & Plajer, I. C. (2024). Learning about Growing Neural Cellular Automata. IEEE Access.

**Library Credits**
- [Numpy](https://numpy.org/): Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.
- [OpenCV](https://opencv.org/): Bradski, G., & Kaehler, A. (2000). OpenCV. Dr. Dobb’s journal of software tools, 3(2).
- [PyTorch](https://pytorch.org/): Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.
## License
This code is made publicly available under the GNU GPLv3 license.
