import pickle
import pathlib
import builtins

def start(idx):
    pass


def persistent_iteration(iterable):
    global iterator
    if _now_recovering(filename):
        return load(filename)["iterator"]
    iterator = iter(iterable)
    for v in iterator:
        yield v


def load(filename):
    global iterator
    if _now_recovering(filename):
        with open(filename, "rb") as snapshot:
            dict_res = pickle.load(snapshot)
            iterator = dict_res.pop("@")
            assert isinstance(dict_res, dict)
            return dict_res
    return {}


def _recover(snapshot):
    with open("snapshot.pkl", "rb") as snapshot:
        return pickle.load(snapshot)


def commit(**kwargs):
    with open("snapshot.pkl", "wb") as snapshot:
        kwargs = {"@": iterator, **kwargs}
        pickle.dump(kwargs, snapshot)


def _mark(S):
    pass


def _mark_shallow(S):
    pass


def _now_recovering(filename):
    return pathlib.Path(filename).exists()


def _unmark(S):
    pass
