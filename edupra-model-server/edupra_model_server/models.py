from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
from pydantic import BaseModel
from gym_backgammon.envs.backgammon import Backgammon, BackgammonState
from gym_backgammon.envs.backgammon import WHITE, BLACK
from edupra_core.agents import Agent


class Request(BaseModel):
    command: str


class BaseResponse(BaseModel):
    message: str


class Response(BaseResponse):
    state: BackgammonState
    actions: Sequence[Any]


@dataclass
class GameState:
    agents: Dict
    env: Backgammon
    agent: Optional[Agent] = None
    first_roll: Optional[Tuple[int, int]] = None
    wins: Dict = field(default_factory=lambda: {WHITE: 0, BLACK: 0})
    roll: Optional[Tuple[int, int]] = None
    game_started: bool = False
    game_finished: bool = False
    last_commands: Sequence[str] = field(default_factory=lambda: [])
