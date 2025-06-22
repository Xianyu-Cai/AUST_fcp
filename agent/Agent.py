from agent.Base_Agent import Base_Agent
from agent.Role_Manager import Role_Manager
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # 定义机器人类型
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # 初始化基础代理
        # 参数: 服务器IP, 代理端口, 监控端口, 球衣号码, 机器人类型, 队伍名称, 启用日志, 启用绘图, 比赛模式修正, 等待服务器, 听回调
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-正常, 1-起身, 2-踢球
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # fat proxy的过滤行走参数
        self.role_manager = Role_Manager(self.world)

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # 初始阵型


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # 复制位置列表 
        self.state = 0

        # 通过将玩家向后移动来避免中心圆
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # 传送到初始位置，面向坐标(0,0)
        else:
            if self.fat_proxy_cmd is None: # 正常行为
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy行为
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # 重置fat proxy行走



    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        走到目标位置

        参数
        ----------
        target_2d : array_like
            绝对坐标中的2D目标
        orientation : float
            躯干的绝对或相对方向，以度为单位
            设置为None以朝向目标（忽略is_orientation_absolute）
        is_orientation_absolute : bool
            如果方向相对于场地为True，如果相对于机器人躯干为False
        avoid_obstacles : bool
            使用路径规划避免障碍物为True（如果此函数在每个模拟周期多次调用，可能需要减少timeout参数）
        priority_unums : list
            需要避免的队友列表（因为他们的角色更重要）
        is_aggressive : bool
            如果为True，则减少对手的安全边际
        timeout : float
            将路径规划限制为最大持续时间（以微秒为单位）    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy行为
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # 忽略障碍物
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # 参数: 目标, 是否绝对目标, 方向, 是否绝对方向, 距离


    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        走到球并踢球

        参数
        ----------
        kick_direction : float
            踢球方向，相对于场地的度数
        kick_distance : float
            踢球距离，以米为单位
        abort : bool
            为True时中止。
            方法在成功中止时返回True，这在机器人对齐时是立即的。
            然而，如果在中踢期间请求中止，则延迟到踢球完成。
        avoid_pass_command : bool
            当为False时，当至少一个对手靠近球时将使用传球命令
            
        返回
        -------
        finished : bool
            如果行为完成或成功中止，则返回True。
        '''

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # 正常行为
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick没有踢球距离控制
        else: # fat proxy行为
            return self.fat_proxy_kick()


    def think_and_send(self):
        w = self.world
        r = self.world.robot  
        my_head_pos_2d = r.loc_head_position[:2]
        my_ori = r.imu_torso_orientation
        ball_2d = w.ball_abs_pos[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = M.vector_angle(ball_vec)
        ball_dist = np.linalg.norm(ball_vec)
        ball_sq_dist = ball_dist * ball_dist # 用于更快的比较
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
        behavior = self.behavior
        goal_dir = M.target_abs_angle(ball_2d,(15.05,0))
        path_draw_options = self.path_manager.draw_options
        PM = w.play_mode
        PM_GROUP = w.play_mode_group

        #--------------------------------------- 1. 预处理

        slow_ball_pos = w.get_predicted_ball_pos(0.5) # 当球速<= 0.5 m/s时的预测未来2D球位置

        # 队友（包括自己）和慢球之间的平方距离列表（某些条件下平方距离设置为1000）
        teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # 队友和球之间的平方距离
                                  if p.state_last_update != 0 and (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                  else 1000 # 如果队友不存在，或其状态信息不最近（360 ms），或已倒下，则强制大距离
                                  for p in w.teammates ]

        # 对手和慢球之间的平方距离列表（某些条件下平方距离设置为1000）
        opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)  # 对手和球之间的平方距离
                                  if p.state_last_update != 0 and w.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                  else 1000 # 如果对手不存在，或其状态信息不最近（360 ms），或已倒下，则强制大距离
                                  for p in w.opponents ]

        min_teammate_ball_sq_dist = min(teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(min_teammate_ball_sq_dist)   # 球和最近队友之间的距离
        self.min_opponent_ball_dist = math.sqrt(min(opponents_ball_sq_dist)) # 球和最近对手之间的距离

        active_player_unum = teammates_ball_sq_dist.index(min_teammate_ball_sq_dist) + 1
        self.role_manager.update_role(active_player_unum)


        #--------------------------------------- 2. 决定行动
        if PM == w.M_GAME_OVER:
            pass
        elif PM_GROUP == w.MG_ACTIVE_BEAM:
            self.beam()
        elif PM_GROUP == w.MG_PASSIVE_BEAM:
            self.beam(True) # 避免中心圆
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1 # 如果起身行为完成，则返回正常状态
        elif PM == w.M_OUR_KICKOFF:
            if r.unum == 9:
                self.kick(120,3) # 当PM不是Play On时不需要改变状态
            else:
                self.move(self.init_pos, orientation=ball_dir) # 原地行走
        elif PM == w.M_THEIR_KICKOFF:
            self.move(self.init_pos, orientation=ball_dir) # 原地行走
        elif active_player_unum != r.unum: # 我不是活跃玩家
            if r.unum == 1: # 我是守门员
                self.move(self.init_pos, orientation=ball_dir) # 原地行走 
            else:
                # 根据球的位置计算基本阵型位置
                new_x = max(0.5,(ball_2d[0]+15)/15) * (self.init_pos[0]+15) - 15
                if self.min_teammate_ball_dist < self.min_opponent_ball_dist:
                    new_x = min(new_x + 3.5, 13) # 如果队伍控球则前进
                self.move((new_x,self.init_pos[1]), orientation=ball_dir, priority_unums=[active_player_unum])

        else: # 我是活跃玩家
            path_draw_options(enable_obstacles=True, enable_path=True, use_team_drawing_channel=True) # 为活跃玩家启用路径绘图（如果self.enable_draw为False则忽略）
            enable_pass_command = (PM == w.M_PLAY_ON and ball_2d[0]<6)

            if r.unum == 1 and PM_GROUP == w.MG_THEIR_KICK: # 对方踢球时的守门员
                self.move(self.init_pos, orientation=ball_dir) # 原地行走 
            if PM == w.M_OUR_CORNER_KICK:
                self.kick( -np.sign(ball_2d[1])*95, 5.5) # 将球踢向对方球门前的空间
                # 当PM不是Play On时不需要改变状态
            elif self.min_opponent_ball_dist + 0.5 < self.min_teammate_ball_dist: # 如果对手明显更接近球则防守
                if self.state == 2: # 在踢球时中止
                    self.state = 0 if self.kick(abort=True) else 2
                else: # 移动到球的位置，但将自己定位在球和我们球门之间
                    self.move(slow_ball_pos + M.normalize_vec((-16,0) - slow_ball_pos) * 0.2, is_aggressive=True)
            else:
                if self.min_opponent_ball_dist < 1.0:
                    others = [(i,d) for i,d in enumerate(teammates_ball_sq_dist) if i != r.unum-1]
                    if others:
                        idx,dist_val = min(others, key=lambda x:x[1])
                        teammate = w.teammates[idx]
                        if teammate.state_abs_pos is not None and dist_val < min_teammate_ball_sq_dist:
                            behavior.execute("Pass", teammate.state_abs_pos[:2])
                            self.state = 2
                        else:
                            self.state = 0 if self.kick(goal_dir,9,False,enable_pass_command) else 2
                    else:
                        self.state = 0 if self.kick(goal_dir,9,False,enable_pass_command) else 2
                else:
                    self.state = 0 if self.kick(goal_dir,9,False,enable_pass_command) else 2

            path_draw_options(enable_obstacles=False, enable_path=False) # 禁用路径绘图

        #--------------------------------------- 3. 广播
        self.radio.broadcast()

        #--------------------------------------- 4. 发送至服务器
        if self.fat_proxy_cmd is None: # 正常行为
            self.scom.commit_and_send( r.get_command() )
        else: # fat proxy行为
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""

        #---------------------- 调试注释
        if self.enable_draw: 
            d = w.draw
            if active_player_unum == r.unum:
                d.point(slow_ball_pos, 3, d.Color.pink, "status", False) # 当球速<= 0.5 m/s时的预测未来2D球位置
                d.point(w.ball_2d_pred_pos[-1], 5, d.Color.pink, "status", False) # 最后的球预测
                d.annotation((*my_head_pos_2d, 0.6), "I've got it!" , d.Color.yellow, "status")
            else:
                d.clear("status")


    #--------------------------------------- Fat proxy辅助方法


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy踢球参数: 力量 [0,10]; 相对水平角度 [-180,180]; 垂直角度 [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # 重置fat proxy行走
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # 忽略障碍物
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")
