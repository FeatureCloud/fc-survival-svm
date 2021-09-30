class Exchanged:
    pass


class Signal(Exchanged):
    pass


class SyncSignal(Signal):
    pass


class SyncSignalClient(SyncSignal):
    pass


class SyncSignalCoordinator(SyncSignal):
    pass


class ConfigFinished(SyncSignalClient):
    pass
