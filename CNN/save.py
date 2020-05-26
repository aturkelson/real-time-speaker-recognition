import pickle

def dump(data, fname):
    """
    Dumps the model object to file.

    Args:
        fname (int): Name of the model file
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f, -1)

def load(fname):
    """
    Loads a model object from a dump file.

    Args:
        fname (string): Dump file of the model

    Returns:
        ModelInterface: Model object
    """
    with open(fname, 'rb') as f:
        R = pickle.load(f)
        return R