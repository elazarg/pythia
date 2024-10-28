from checkpoint import persist


def run() -> None:
    """Trivial baseline"""
    for i in range(100):  # type: int
        persist.self_coredump()
        pass


if __name__ == "__main__":
    run()
