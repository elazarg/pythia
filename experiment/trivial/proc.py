from checkpoint import persist


def run() -> None:
    """Trivial baseline"""
    with persist.snapshotter("trivial") as self_coredump:
        for i in range(20):  # type: int
            self_coredump()
            pass


if __name__ == "__main__":
    run()
