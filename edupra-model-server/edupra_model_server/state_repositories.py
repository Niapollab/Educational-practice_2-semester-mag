from typing import Dict
from .models import GameState
from abc import ABC, abstractmethod


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
