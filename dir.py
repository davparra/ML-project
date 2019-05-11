import os

def get_base_dir():
    return os.path.join(os.getcwd(), 'data')

def get_train_dir():
    return os.path.join(get_base_dir(), 'training')

def get_train_case_dir(case):
    return os.path.join(get_train_dir(), case)

def get_val_dir():
    return os.path.join(get_base_dir(), 'validation')

def get_val_case_dir(case):
    return os.path.join(get_val_dir(), case)