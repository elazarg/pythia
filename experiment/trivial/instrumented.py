from checkpoint import persist


def run() -> None:
    """Trivial baseline"""
    with persist.Loader(__file__, locals()) as transaction:
        if transaction:
            [i] = transaction.move()
        for i in transaction.iterate(range(100)):  # type: int
            pass
            transaction.commit(i)


if __name__ == "__main__":
    run()
