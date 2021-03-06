

def polynomial_decay(global_step,
    init_learning_rate = 1e-3,
    min_learning_rate = 1e-5,
    total_epochs = 200):
    return init_learning_rate - \
            global_step * init_learning_rate / total_epochs + \
            min_learning_rate