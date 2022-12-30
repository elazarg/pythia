import pickle


def start(idx):
    pass


def commit():
    pass


def _recover(snapshot):
    with open("snapshot.pkl", "rb") as snapshot:
        return pickle.load(snapshot)


def _commit(*args):
    with open("snapshot.pkl", "wb") as snapshot:
        pickle.dump(args, snapshot)


def _mark(S):
    pass


def _mark_shallow(S):
    pass


def _now_recovering():
    pass


def _unmark(S):
    pass
