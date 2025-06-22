import types, sys
import pytest
np = pytest.importorskip('numpy')

sys.modules['cpp.localization'] = types.ModuleType('localization')
sys.modules['cpp.localization'].localization = None
sys.modules['cpp.ball_predictor'] = types.ModuleType('ball_predictor')
sys.modules['cpp.ball_predictor'].ball_predictor = None
mock_a_star = types.ModuleType('a_star')

def compute(params):
    import numpy as np
    return np.array([params[0], params[1], params[4], params[5], 3, 0], np.float32)
mock_a_star.compute = compute
sys.modules['cpp.a_star'] = mock_a_star

from logs.Logger import Logger
from world.World import World
from world.commons.Path_Manager import Path_Manager


def test_get_path_to_target():
    w = World(0, 'T', 1, True, False, Logger(False), 'localhost')
    w.robot.loc_head_position[:] = [0,0,0]
    pm = Path_Manager(w)
    target = np.array([1.0, 0.0])
    pos, ori, dist = pm.get_path_to_target(target, timeout=1000)
    assert np.allclose(pos, target)
    assert pytest.approx(ori, abs=1e-3) == 0
    assert dist >= 1.0
