from tools import read_file


def train_and_evaluate(path,
                       filename,
                       model,
                       trainer,
                       evaluator,
                       win_size=None,
                       epochs=20,
                       stride=1, 
                       data=None):
    """
    Read dataset from filename, train model and evaluate.
    trainer and evaluator should be instantiated by the caller.
    """
    if data is not None:
        data_train, data, labels = data
    else:
        data_train, data, labels = read_file(path, filename)

    if win_size is None:
        win_size = trainer.win_size
    else:
        trainer.win_size = win_size

    trainer.train(model, data_train, epochs)

    return evaluator.evaluate(data, labels, model, win_size, stride=stride)


def compare_reconstruction(path,
                           filename,
                           model,
                           trainer,
                           evaluator1,
                           evaluator2,
                           win_size=None,
                           epochs=20):

    data_train, data, _ = read_file(path, filename)

    if win_size is None:
        win_size = trainer.win_size
    else:
        trainer.win_size = win_size

    trainer.train(model, data_train, epochs)

    reconstruction1 = evaluator1.reconstruction_error(data, model, win_size)
    reconstruction2 = evaluator2.reconstruction_error(data, model, win_size)

    return reconstruction1, reconstruction2
