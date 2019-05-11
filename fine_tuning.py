
def fine_tune_training():
        pass

def freeze(model, n):
    # Freeze all the layers before the `n` layer
    for layer in model.layers[:n]:
        layer.trainable = False

def freeze_all(model):
    # Freeze all the layers
    for layer in model.layers:
        layer.trainable = False

def unfreeze(model, n):
    # unfreeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:n]:
        layer.trainable = True

def unfreeze_all(model):
    # Freeze all the layers
    for layer in model.layers:
        layer.trainable = True

def unfreeze_batch_normalization(model):
    #TODO: try fine tuned model retraining batch normalization
    pass