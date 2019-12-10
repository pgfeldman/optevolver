import tensorflow as tf
import os
import shutil

def clear_directory(target_dir:str):
    """ Empties a target directory of all files and nested folders
        Parameters
        ----------
        target_dir:str
            The fully qualified or relative name of the target directory
    """
    with os.scandir(target_dir) as entries:
        for entry in entries:
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)

def del_dir(target_dir:str) -> bool:
    """ Empties  target directory of all files and nested folders, then deletes the directory. Returns success
        Parameters
        ----------
        target_dir:str
            The fully qualified or relative name of the target directory
    """
    try:
        clear_directory(target_dir)
        os.rmdir(target_dir)
        return True
    except OSError as e:
        print("ModelWriter.del_dir(): {} - {}".format(e.filename, e.strerror))
        return False

def create_dir(dir_name:str) -> bool:
    """ Creates a directory. Returns success
        Parameters
        ----------
        dir_name:str
            The name of the directory

    """
    try:
        os.mkdir(dir_name)
        return True
    except OSError as e:
        print("ModelWriter.create_dir(): {} - {}".format(e.filename, e.strerror))
        return False

def move_dir(old:str, new:str) -> bool:
    """ Moves a directory. Returns success
        Parameters
        ----------
        old : str
            The old name of the directory
        new : str
            The new name of the directory

    """
    try:
        os.rename(old, new)
        return True
    except OSError as e:
        print("ModelWriter.move_dir(): {} - {}".format(e.filename, e.strerror))
        return False

def write_model(path:str, modelname:str, model:tf.keras.Model) -> bool:
    """ Writes out a TF2.0 model structure and weights. Returns success
        Parameters
        ----------
        path:str
            The name of the root directory
        modelname:str
            The name of the model
        model:tf.keras.Model
            The model to write

    """
    p = os.getcwd()
    os.chdir(path)
    model.save(modelname)
    os.chdir(p)
    return True