import pickle
import pathlib
import hashlib


class Iter:
    def __init__(self, iterable):
        self.iterator = iter(iterable)

    def __iter__(self):
        return self.iterator


class Loader:
    def __init__(self, module_filename):

        with open(module_filename, 'rb') as f:
            m = hashlib.md5()
            m.update(f.read())
            h = m.hexdigest()[:6]
            name = module_filename.split('.')[-2]

        self.filename = f'{name}-{h}.pickle'
        self.iterator = None
        self.version = 0

    def __enter__(self) -> tuple[object, ...]:
        if self._now_recovering():
            print("Recovering from snapshot")
            with open(self.filename, "rb") as snapshot:
                version, values, self.iterator = pickle.load(snapshot)
                print(f'Loaded {version=}: {values}')
                return values
        return ()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print('Finished successfully')
            pathlib.Path(self.filename).unlink()

    def iterate(self, iterable):
        if self.iterator is None:
            self.iterator = Iter(iterable)
        return self.iterator

    def commit(self, *args):
        self.version += 1

        temp_filename = f'{self.filename}.tmp'
        with open(temp_filename, "wb") as snapshot:
            # fix: do this transactionally
            pickle.dump((self.version, args, self.iterator), snapshot)

        pathlib.Path(self.filename).unlink(missing_ok=True)
        pathlib.Path(temp_filename).rename(self.filename)

    def _now_recovering(self):
        return pathlib.Path(self.filename).exists()
