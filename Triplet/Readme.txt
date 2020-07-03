Application of deep metric learning based on Von Mises-fisher distribution loss.

########################### package ######################
- we used pytorch framework 

########################### Papers #######################
- VMF:     https://arxiv.org/abs/1802.09662

########################### Paramteres: Config file #######################
Facnet_config = {
    "Epochs" : 1,              # Number of epochs
    "num_iter" : 1,            # Number of iteration on one single minibatch (we can iterate more than once)
    "max_batch" : 1000,        # Maximum number of batches per epock
    "learning_rate" : 10**-3,  # learning rate
    "alpha" : 0.2,             # Margin alpha for triplet loss
    "triplet_K" : 50,          # K of K-nearest neighbors for generating triplets
    "K" : 15,                  # K for K-nearest neighbors
    "list_classes" : [0,1],    # List of classes
    "batch_size" : 500,        # batch size
    "optimizer_flag" : 'Adam', # Optimizer
    "width" : 2,               # width of wide residual block
}

########################### Main #######################
images_train/images_test shape = number of samples*3*32*32:
  - 3 for 3 channels as RGB.
  - 32*32 is the image sample diemnsion.

label_train/label_test shape = (number of samples,).

- Upload train/test data then run the main file.

