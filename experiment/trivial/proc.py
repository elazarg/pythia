from checkpoint import persist


def run() -> None:
    """Trivial baseline"""
    with persist.snapshotter() as self_coredump:
        for i in range(100):  # type: int
            self_coredump("trivial")
            pass


if __name__ == "__main__":
    run()
