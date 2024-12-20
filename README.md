# MNIST Classification with PyTorch

This notebook implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch.

## Model Architecture

The model uses a custom CNN architecture with the following key features:

- Input Block: Initial convolution layer with 8 channels
- Multiple Convolution Blocks with increasing channels (8->12->10->12->16)
- Batch Normalization after each convolution layer
- Dropout (value: 0.01) for regularization
- Global Average Pooling (GAP) layer before final output
- Total Parameters: 7,776

## Key Components

1. **Data Augmentation**:
   - Random rotation (-7° to 7°)
   - Normalization with mean=0.1307, std=0.3081

2. **Training Configuration**:
   - Optimizer: SGD with Nesterov momentum
   - Learning Rate: 0.01 with StepLR scheduler
   - Batch Size: 128 (with CUDA)
   - Weight Decay: 1e-4 for regularization

3. **Results**:
   - Best Training Accuracy: 99.28%
   - Best Test Accuracy: 99.44%
   - Consistent performance around 99.40% after epoch 10

## Requirements

- PyTorch
- torchvision
- tqdm
- torchsummary

## Features

- CUDA support for GPU acceleration
- Progress bars using tqdm
- Model summary visualization
- Learning rate scheduling
- Comprehensive training and testing loops

## Performance Analysis

The model achieves excellent performance with:
- High accuracy (>99.3%)
- Good consistency in later epochs
- Minimal overfitting due to effective regularization
- Fast convergence (reaches >98% accuracy within first few epochs)

## Usage

The notebook is self-contained and can be run end-to-end. It includes:
1. Data loading and preprocessing
2. Model definition
3. Training loop
4. Testing and evaluation
5. Performance visualization

Simply run all cells in sequence to train and evaluate the model.

## Steps taken to achieve

###Step-1:

####Target:

Make a lighter model under 8k parameters from the original 13,808 parameters

####Results:
Parameters: 7,776<br/>
Best Train Accuracy: 99.00<br/>
Best Test Accuracy: 99.32% (13th Epoch), 99.31% (14th Epoch)<br/>

####Analysis:<br/>
Find a good foundational model architecture.<br/>
Now have to improve the consistency and improve the train accuracy.

---------------------------------------------------------------------------------------------------------------------

###Step-2:

####Target:

Added rotation augmentation to increase the test accuracy more and make the model more robust.

####Results:
Parameters: 7,776<br/>
Best Train Accuracy: 98.72<br/>
Best Test Accuracy: 99.34% (14th Epoch)<br/>

####Analysis:<br/>
Find a good foundational model architecture which is robust after adding augmentation.<br/>
Now have to improve the consistency and improve the train accuracy.
---------------------------------------------------------------------------------------------------------------------
###Step-3:

####Target:

Added weight_decay in optimizer to penalize large weights and reduced the Dropout value to reduce the variability of the model since the model was not learning better than before(Means training accuracy was not improving)<br/>

####Results:
Parameters: 7,776<br/>
Best Train Accuracy: 99.30<br/>
Best Test Accuracy: 99.44%% (10th and consistent since then)<br/>

####Analysis:<br/>
The model seems to have improved consistency and improved the train accuracy as well.



## Final Logs
EPOCH: 0
  0%|          | 0/469 [00:00<?, ?it/s]
Loss=0.08702781051397324 Batch_id=468 Accuracy=89.31: 100%|██████████| 469/469 [00:09<00:00, 51.85it/s]

Test set: Average loss: 0.0914, Accuracy: 9747/10000 (97.47%)

EPOCH: 1
Loss=0.03857508301734924 Batch_id=468 Accuracy=97.77: 100%|██████████| 469/469 [00:05<00:00, 88.22it/s] 

Test set: Average loss: 0.0494, Accuracy: 9873/10000 (98.73%)

EPOCH: 2
Loss=0.0715828463435173 Batch_id=468 Accuracy=98.32: 100%|██████████| 469/469 [00:05<00:00, 88.96it/s]  

Test set: Average loss: 0.0396, Accuracy: 9883/10000 (98.83%)

EPOCH: 3
Loss=0.014542591758072376 Batch_id=468 Accuracy=98.55: 100%|██████████| 469/469 [00:05<00:00, 88.62it/s]

Test set: Average loss: 0.0328, Accuracy: 9905/10000 (99.05%)

EPOCH: 4
Loss=0.03415935859084129 Batch_id=468 Accuracy=98.69: 100%|██████████| 469/469 [00:05<00:00, 87.58it/s]  

Test set: Average loss: 0.0294, Accuracy: 9917/10000 (99.17%)

EPOCH: 5
Loss=0.021290138363838196 Batch_id=468 Accuracy=98.79: 100%|██████████| 469/469 [00:05<00:00, 87.47it/s] 

Test set: Average loss: 0.0311, Accuracy: 9918/10000 (99.18%)

EPOCH: 6
Loss=0.03946084901690483 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:05<00:00, 88.21it/s]  

Test set: Average loss: 0.0236, Accuracy: 9932/10000 (99.32%)

EPOCH: 7
Loss=0.04197080805897713 Batch_id=468 Accuracy=99.16: 100%|██████████| 469/469 [00:05<00:00, 90.96it/s]  

Test set: Average loss: 0.0222, Accuracy: 9941/10000 (99.41%)

EPOCH: 8
Loss=0.01661953516304493 Batch_id=468 Accuracy=99.18: 100%|██████████| 469/469 [00:05<00:00, 89.40it/s]  

Test set: Average loss: 0.0221, Accuracy: 9935/10000 (99.35%)

EPOCH: 9
Loss=0.008231011219322681 Batch_id=468 Accuracy=99.19: 100%|██████████| 469/469 [00:05<00:00, 89.27it/s] 

Test set: Average loss: 0.0220, Accuracy: 9938/10000 (99.38%)

EPOCH: 10
Loss=0.006600108463317156 Batch_id=468 Accuracy=99.16: 100%|██████████| 469/469 [00:05<00:00, 89.88it/s] 

Test set: Average loss: 0.0221, Accuracy: 9944/10000 (99.44%)

EPOCH: 11
Loss=0.013559146784245968 Batch_id=468 Accuracy=99.23: 100%|██████████| 469/469 [00:05<00:00, 86.91it/s] 

Test set: Average loss: 0.0213, Accuracy: 9939/10000 (99.39%)

EPOCH: 12
Loss=0.005306202918291092 Batch_id=468 Accuracy=99.27: 100%|██████████| 469/469 [00:05<00:00, 91.81it/s] 

Test set: Average loss: 0.0218, Accuracy: 9940/10000 (99.40%)

EPOCH: 13
Loss=0.007453827187418938 Batch_id=468 Accuracy=99.20: 100%|██████████| 469/469 [00:05<00:00, 91.58it/s] 

Test set: Average loss: 0.0214, Accuracy: 9938/10000 (99.38%)

EPOCH: 14
Loss=0.029884198680520058 Batch_id=468 Accuracy=99.30: 100%|██████████| 469/469 [00:05<00:00, 89.47it/s] 

Test set: Average loss: 0.0214, Accuracy: 9940/10000 (99.40%)


