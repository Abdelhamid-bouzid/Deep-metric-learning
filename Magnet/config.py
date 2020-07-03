
Facnet_config = {
    "Epochs" : 500,               # Number of epochs
    "learning_rate" : 10**-4,     # learning rate
    "epsilon" : 1e-8,             # epsilon to avoid 0 in denum in loss function
    "alpha" : 1,                  # Margin alpha for magnet loss
    "nb_clusters" : [30,120],     # Number of clusters per class
    "M" : 16,                     # Number of clusters present in a mini-batch
    "D" : 8,                      # Number of smaples selected from each cluster in the mini-batch
    "K" : 15,                     # K for K-nearest neighbors
    "L" : 3,                      # L number of nearest clusters used to predict the label of a given query (instead of KNN: magnet evaluation)
    "nb_batches" : 30,            
    "list_classes" : [0,1],       # List of classes
    "batch_size" : 32,            # mini batch size to forward the data (out of training)
    "optimizer_flag" : 'Adam',    # Optimizer
    "width" : 2,                  # width of wide residual block
}

