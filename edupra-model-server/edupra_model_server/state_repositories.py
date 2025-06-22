from .models import GameState
from abc import ABC, abstractmethod
from datetime import timedelta
from pickle import dumps, loads
from redis import Redis
from typing import Dict


class StateRepository(ABC):
    @abstractmethod
    def __getitem__(self, uid: str) -> GameState:
        pass

    @abstractmethod
    def __setitem__(self, uid: str, state: GameState) -> None:
        pass

    @abstractmethod
    def __contains__(self, uid: str) -> bool:
        pass


class DictStateRepository(StateRepository):
    _states: Dict

    def __init__(self) -> None:
        self._states = {}

    def __getitem__(self, uid: str) -> GameState:
        return self._states[uid]

    def __setitem__(self, uid: str, state: GameState) -> None:
        self._states[uid] = state

    def __contains__(self, uid: str) -> bool:
        return uid in self._states


class RedisStateRepository(StateRepository):
    _client: Redis

    def __init__(self, url: str) -> None:
        self._client = Redis.from_url(url)

    def __getitem__(self, uid: str) -> GameState:
        return loads(self._client.get(uid))

    def __setitem__(self, uid: str, state: GameState) -> None:
        self._client.setex(uid, timedelta(hours=6), dumps(state))

    def __contains__(self, uid: str) -> bool:
        return self._client.exists(uid)

    def __enter__(self) -> "RedisStateRepository":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
