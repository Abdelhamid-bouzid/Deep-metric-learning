
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
