from checkpoint import persist


def run() -> None:
    """Trivial baseline"""
    with persist.SimpleTcpClient("trivial") as client:
        for i in client.iterate(range(20)):  # type: int
            pass
            client.commit()


if __name__ == "__main__":
    run()
