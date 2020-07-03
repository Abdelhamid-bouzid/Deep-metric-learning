

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