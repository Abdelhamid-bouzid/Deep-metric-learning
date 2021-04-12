
config = {
    "iteration"      : 500000,        # Number of epochs
    "learning_rate"  : 1e-4,     # learning rate
    "lr_decay_factor": 0.2,
    "lr_decay_iter"  : 400000,
    "Con"            : 5,         # Concentration of vmf distribution
    "batch_size"     : 32,         # batch size
    "optimizer_flag" : 'Adam',     # Optimizer 
    "width"          : 2,          # width of wide residual block
    
    "transform"      : [True, True, True], # flip, rnd crop, gaussian noise
    
    "ema_factor"     : 0.95,
    "consis_coef"    : 100,
    "alpha"          : 0.1,
    
    "warmup"         : 20000,
    "topk"           : 8,
    
    "test_model_cycel" :1500,
    
    "mean_dir_cycle"   : 1500,
    "list_classes"     : [0,1,2,3,4,5,6,7,8,9],
    "nb_clusters"      : 50,
}
