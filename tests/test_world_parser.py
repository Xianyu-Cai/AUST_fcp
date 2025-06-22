import types, sys
import pytest
np = pytest.importorskip('numpy')

sys.modules['cpp.localization'] = types.ModuleType('localization')
sys.modules['cpp.localization'].localization = None
sys.modules['cpp.ball_predictor'] = types.ModuleType('ball_predictor')
sys.modules['cpp.ball_predictor'].ball_predictor = None
sys.modules['cpp.a_star'] = types.ModuleType('a_star')
sys.modules['cpp.a_star'].compute = lambda p: None

from logs.Logger import Logger
from world.World import World
from communication.World_Parser import World_Parser


def test_world_parser_basic():
    w = World(0, 'T', 1, True, False, Logger(False), 'localhost')
    wp = World_Parser(w, lambda *args: None)
    msg = b'(time (now 0))(GS (unum 1)(team left)(sl 0)(sr 0)(t 0)(pm BeforeKickOff))'
    wp.parse(msg)
    assert w.step == 1
    assert w.play_mode == World.M_BEFORE_KICKOFF
