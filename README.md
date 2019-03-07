# Custom_V-Net
Extensions and modifications implemented to extend the original V-Net.
All the extensions and modificationscan be disabled and adjusted in the 'main.py' file.

Added features:
- Adam solver
- Validation phases
- Probability maps as input
- k-fold cross-validation
- Save snapshot of the iteration with the best validation loss
- Auto-Context model

New hyperparameters that can be adjusted in the 'main.py' file:
- weight decay
- momentum parameter of the solver
- number of bins for cross-validation
- interval between validation phases
- number of images for validation phase
- disable preprocessing
- probability for data augmentation


