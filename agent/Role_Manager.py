from world.World import World

class Role_Manager:
    GOALKEEPER = 'goalkeeper'
    STRIKER = 'striker'
    SUPPORT = 'support'

    def __init__(self, world: World) -> None:
        self.world = world

    def update_role(self, active_unum: int) -> str:
        r = self.world.robot
        if r.unum == 1:
            r.current_role = Role_Manager.GOALKEEPER
        elif r.unum == active_unum:
            r.current_role = Role_Manager.STRIKER
        else:
            r.current_role = Role_Manager.SUPPORT
        return r.current_role
