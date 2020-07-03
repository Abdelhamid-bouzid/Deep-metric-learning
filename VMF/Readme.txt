Application of deep metric learning based on Von Mises-fisher distribution loss.

########################### package ######################
- we used pytorch framework 

########################### Papers #######################
- VMF:     https://arxiv.org/abs/1802.09662

########################### Paramteres: Config file #######################
Facnet_config = {
    "Epochs" : 300,               # Number of epochs
    "learning_rate" : 10**-5,     # learning rate
    "Con" : 15,                   # Concentration of vmf distribution
    "K" : 15,                     # K for K-nearest neighbors
    "list_classes" : [0,1],       # List of classes
    "batch_size" : 32,            # batch size
    "optimizer_flag" : 'Adam',    # Optimizer 
    "width" : 2,                  # width of wide residual block
}

########################### Main #######################
images_train/images_test shape = number of samples*3*32*32:
  - 3 for 3 channels as RGB.
  - 32*32 is the image sample diemnsion.

label_train/label_test shape = (number of samples,).

- Upload train/test data then run the main file.

