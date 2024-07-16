# modular-nca
**An experimental Python-based framework for training modular neural cellular automata (MNCAs).**

 ![Neural network visualisation](https://github.com/paveworkshop/modular-nca/blob/main/previews/neural-network.png)  
_The neural network embedded inside each simulation 'pixel', which mimics a gene regulatory network._
 
**A petri dish for self-taught growth dynamics**  

![Neural network visualisation](https://github.com/paveworkshop/modular-nca/blob/main/previews/hex-leaf-mono-7-0-64-16-9241-3826-1721073373.gif)  
The cellular automata 'substrate' is composed of 3 visible channels (rgb), 1 alpha channel and 12 hidden channels (h0-h12) which the model decides how to use.  

_In this discarded experiment, the NCA learnt to 'cheat' the task of growing a leaf by using the walls of the substrate._

## Preview
A modular NCA trained on a simplified leaf image.  

**Training sequence**  
 ![Leaf training result](https://github.com/paveworkshop/modular-nca/blob/main/training_datasets/leaf-mono-thumbnail.png)
 
**Training result**  
![Leaf training sequence](https://github.com/paveworkshop/modular-nca/blob/main/previews/hex-leaf-mono-1-0-96-16-9241-1248-1721141463.gif) 

## Contents
- [Scripts](#Scripts)
- [Usage](#Usage)
- [Credits](#Credits)
- [License](#License)

## Scripts
- [train_nca.py]()
- [eval_nca.py](https://ds.lis.2i2c.cloud/hub/user-redirect/lab/tree/eval_nca.ipynb)
  
## Usage
Click one of the script links above to open a Jupyter notebook session.


## Credits
The NCA architecture was based on [this](https://distill.pub/2020/growing-ca/) paper:  
Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020). Growing neural cellular automata. Distill, 5(2), e23.

**Library Credits**
- [Numpy](https://numpy.org/): Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.
- [OpenCV](https://opencv.org/): Bradski, G., & Kaehler, A. (2000). OpenCV. Dr. Dobb’s journal of software tools, 3(2).
- [PyTorch](https://pytorch.org/): Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

## License
This code is made publicly available under the GNU GPLv3 license.
