from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M

class Pass:
    def __init__(self, base_agent: Base_Agent) -> None:
        self.agent = base_agent
        self.behavior = base_agent.behavior
        self.world = base_agent.world
        self.description = "Pass ball to teammate"
        self.auto_head = True

    def execute(self, reset, teammate_pos):
        direction = M.target_abs_angle(self.world.ball_abs_pos[:2], teammate_pos)
        if reset:
            self.agent.scom.commit_pass_command()
        return self.behavior.execute_sub_behavior("Basic_Kick", reset, direction, False)

    def is_ready(self):
        return True
