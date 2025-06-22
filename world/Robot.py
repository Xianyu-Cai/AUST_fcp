from collections import deque
from math import atan, pi, sqrt, tan
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Matrix_3x3 import Matrix_3x3
from math_ops.Matrix_4x4 import Matrix_4x4
from world.commons.Body_Part import Body_Part
from world.commons.Joint_Info import Joint_Info
import numpy as np
import xml.etree.ElementTree as xmlp

class Robot():
    STEPTIME = 0.02   # 固定步长时间
    VISUALSTEP = 0.04 # 固定视觉步长时间
    SQ_STEPTIME = STEPTIME * STEPTIME
    GRAVITY = np.array([0,0,-9.81])
    IMU_DECAY = 0.996 # IMU的速度衰减
            
    #------------------ 强制关节/效应器对称性的常量

    MAP_PERCEPTOR_TO_INDEX = {"hj1":0,  "hj2":1,  "llj1":2, "rlj1":3,
                              "llj2":4, "rlj2":5, "llj3":6, "rlj3":7,
                              "llj4":8, "rlj4":9, "llj5":10,"rlj5":11,
                              "llj6":12,"rlj6":13,"laj1":14,"raj1":15,
                              "laj2":16,"raj2":17,"laj3":18,"raj3":19,
                              "laj4":20,"raj4":21,"llj7":22,"rlj7":23 }

    # 修复对称性问题 1a/4 (识别)                                  
    FIX_PERCEPTOR_SET = {'rlj2','rlj6','raj2','laj3','laj4'}
    FIX_INDICES_LIST = [5,13,17,18,20]


    # 推荐的非官方光束高度(接近地面)
    BEAM_HEIGHTS = [0.4, 0.43, 0.4, 0.46, 0.4]


    def __init__(self, unum:int, robot_type:int) -> None:
        robot_xml = "nao"+str(robot_type)+".xml" # 典型的NAO文件名
        self.type = robot_type
        self.beam_height = Robot.BEAM_HEIGHTS[robot_type]
        self.no_of_joints = 24 if robot_type == 4 else 22 

        # 修复对称性问题 1b/4 (识别) 
        self.FIX_EFFECTOR_MASK = np.ones(self.no_of_joints)
        self.FIX_EFFECTOR_MASK[Robot.FIX_INDICES_LIST] = -1

        self.body_parts = dict()    # 键='身体部位名称'(由agent的XML给出),值='Body_Part对象'
        self.unum = unum            # agent的球衣号码
        self.current_role = 'support'  # 当前角色, 由 Role_Manager 更新
        self.gyro = np.zeros(3)     # agent躯干沿三个自由轴的角速度(deg/s)
        self.acc  = np.zeros(3)     # agent躯干沿三个自由轴的适当加速度(m/s2)
        self.frp = dict() # 脚 "lf"/"rf",脚趾 "lf1"/"rf1" 阻力感知器(相对[p]点原点 + [f]力向量)例如 {"lf":(px,py,pz,fx,fy,fz)}
        self.feet_toes_last_touch = {"lf":0,"rf":0,"lf1":0,"rf1":0} # 脚 "lf"/"rf",脚趾 "lf1"/"rf1" 上次接触任何表面的World.time_local_ms
        self.feet_toes_are_touching = {"lf":False,"rf":False,"lf1":False,"rf1":False} # 脚 "lf"/"rf",脚趾 "lf1"/"rf1" 如果在最后接收到的服务器消息中接触则为True
        self.fwd_kinematics_list = None             # 按依赖关系排序的身体部位列表
        self.rel_cart_CoM_position = np.zeros(3)    # 质心位置,相对于头部,在笛卡尔坐标系中(m)

        # 关节变量针对性能进行了优化/数组操作
        self.joints_position =          np.zeros(self.no_of_joints)                      # 关节的角位置 (deg)
        self.joints_speed =             np.zeros(self.no_of_joints)                      # 关节的角速度 (rad/s)
        self.joints_target_speed =      np.zeros(self.no_of_joints)                      # 关节的目标速度 (rad/s)(最大：6.1395 rad/s,见 rcssserver3d/data/rsg/agent/nao/hingejoint.rsg)
        self.joints_target_last_speed = np.zeros(self.no_of_joints)                      # 关节的最后目标速度 (rad/s)(最大：6.1395 rad/s,见 rcssserver3d/data/rsg/agent/nao/hingejoint.rsg)
        self.joints_info =              [None] * self.no_of_joints                       # 关节的常量信息(见类 Joint_Info)
        self.joints_transform =         [Matrix_4x4() for _ in range(self.no_of_joints)] # 关节的变换矩阵

        # 相对于头部的定位变量
        self.loc_head_to_field_transform = Matrix_4x4()  # 从头到场的变换矩阵
        self.loc_field_to_head_transform = Matrix_4x4()  # 从场到头的变换矩阵
        self.loc_rotation_head_to_field = Matrix_3x3()   # 从头到场的旋转矩阵
        self.loc_rotation_field_to_head = Matrix_3x3()   # 从场到头的旋转矩阵
        self.loc_head_position = np.zeros(3)             # 绝对头部位置(m)
        self.loc_head_position_history = deque(maxlen=40)# 绝对头部位置历史(队列,最多包含40个旧位置,间隔0.04s,索引0是前一个位置)
        self.loc_head_velocity = np.zeros(3)             # 绝对头部速度(m/s)(警告：可能有噪声)
        self.loc_head_orientation = 0                    # 头部方向(deg)
        self.loc_is_up_to_date = False                   # 如果不是视觉步长,或者没有足够的元素可见,则为False
        self.loc_last_update = 0                         # 上次更新定位的World.time_local_ms
        self.loc_head_position_last_update = 0           # 上次通过视觉或无线电更新loc_head_position的World.time_local_ms
        self.radio_fallen_state = False                  # 如果(无线电说我们摔倒了)并且(无线电比定位更新得更近)则为True
        self.radio_last_update = 0                       # 上次更新radio_fallen_state(可能还有loc_head_position)的World.time_local_ms

        # 相对于躯干的定位变量
        self.loc_torso_to_field_rotation = Matrix_3x3()  # 从躯干到场的旋转矩阵  
        self.loc_torso_to_field_transform = Matrix_4x4() # 从躯干到场的变换矩阵
        self.loc_torso_roll = 0                          # 躯干滚动 (deg)
        self.loc_torso_pitch = 0                         # 躯干俯仰 (deg) 
        self.loc_torso_orientation = 0                   # 躯干方向 (deg)
        self.loc_torso_inclination = 0                   # 躯干倾斜 (deg)(z轴相对于场z轴的倾斜)
        self.loc_torso_position = np.zeros(3)            # 绝对躯干位置 (m)
        self.loc_torso_velocity = np.zeros(3)            # 绝对躯干速度 (m/s)
        self.loc_torso_acceleration = np.zeros(3)        # 绝对坐标加速度 (m/s2)

        # 其他定位变量
        self.cheat_abs_pos = np.zeros(3)                 # 服务器提供的作弊绝对头部位置 (m)
        self.cheat_ori = 0.0                             # 服务器提供的作弊绝对头部方向 (deg)
        self.loc_CoM_position = np.zeros(3)              # 绝对质心位置 (m)
        self.loc_CoM_velocity = np.zeros(3)              # 绝对质心速度 (m/s)

        # 定位特殊变量
        '''
        self.loc_head_z 通常等同于 self.loc_head_position[2],但有时会有所不同。
        有些情况下,旋转和平移无法计算,
        但z坐标仍然可以通过视觉找到,在这种情况下：
            self.loc_is_up_to_date         为 False
            self.loc_head_z_is_up_to_date  为 True
        它应该用于依赖z作为独立坐标的应用程序,例如
        检测agent是否摔倒,或作为机器学习的观察。
        它永远不应该用于3D变换。
        '''
        self.loc_head_z = 0                     # 绝对头部位置(z) - 见上文解释 (m)
        self.loc_head_z_is_up_to_date = False   # 如果不是视觉步长,或者没有足够的元素可见,则为False
        self.loc_head_z_last_update = 0         # 上次计算loc_head_z的World.time_local_ms
        self.loc_head_z_vel = 0                 # 绝对头部速度(z) (m/s)

        # 定位 + 陀螺仪
        # 这些变量是可靠的。当等待下一个视觉周期时,使用陀螺仪更新旋转
        self.imu_torso_roll = 0                          # 躯干滚动 (deg)                    (来源：定位 + 陀螺仪)
        self.imu_torso_pitch = 0                         # 躯干俯仰 (deg)                    (来源：定位 + 陀螺仪)
        self.imu_torso_orientation = 0                   # 躯干方向 (deg)                    (来源：定位 + 陀螺仪)
        self.imu_torso_inclination = 0                   # 躯干倾斜 (deg)                    (来源：定位 + 陀螺仪)
        self.imu_torso_to_field_rotation = Matrix_3x3()  # 从躯干到场的旋转矩阵        (来源：定位 + 陀螺仪)
        self.imu_last_visual_update = 0                  # 上次使用视觉信息更新IMU数据的World.time_local_ms 

        # 定位 + 陀螺仪 + 加速度计
        # 警告：这些变量不可靠,因为定位方向的小误差会导致
        #          错误的加速度 -> 错误的
        self.imu_weak_torso_to_field_transform = Matrix_4x4() # 从躯干到场的变换矩阵 (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_head_to_field_transform  = Matrix_4x4() # 从头到场的变换矩阵   (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_field_to_head_transform  = Matrix_4x4() # 从场到头的变换矩阵   (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_torso_position = np.zeros(3)        # 绝对躯干位置 (m)                    (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_torso_velocity = np.zeros(3)        # 绝对躯干速度 (m/s)                  (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_torso_acceleration = np.zeros(3)    # 绝对躯干加速度 (m/s2)             (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_torso_next_position = np.zeros(3)   # 下一步估计的绝对位置 (m)    (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_torso_next_velocity = np.zeros(3)   # 下一步估计的绝对速度 (m/s)  (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_CoM_position = np.zeros(3)          # 绝对质心位置 (m)                      (来源：定位 + 陀螺仪 + 加速度计)
        self.imu_weak_CoM_velocity = np.zeros(3)          # 绝对质心速度 (m/s)                    (来源：定位 + 陀螺仪 + 加速度计)


        # 使用显式变量以启用IDE建议
        self.J_HEAD_YAW = 0
        self.J_HEAD_PITCH = 1
        self.J_LLEG_YAW_PITCH = 2
        self.J_RLEG_YAW_PITCH = 3
        self.J_LLEG_ROLL = 4
        self.J_RLEG_ROLL = 5
        self.J_LLEG_PITCH = 6
        self.J_RLEG_PITCH = 7
        self.J_LKNEE = 8
        self.J_RKNEE = 9
        self.J_LFOOT_PITCH = 10
        self.J_RFOOT_PITCH = 11
        self.J_LFOOT_ROLL = 12
        self.J_RFOOT_ROLL = 13
        self.J_LARM_PITCH = 14
        self.J_RARM_PITCH = 15
        self.J_LARM_ROLL = 16
        self.J_RARM_ROLL = 17
        self.J_LELBOW_YAW = 18
        self.J_RELBOW_YAW = 19
        self.J_LELBOW_ROLL = 20
        self.J_RELBOW_ROLL = 21
        self.J_LTOE_PITCH = 22
        self.J_RTOE_PITCH = 23


        #------------------ 解析agentXML

        dir = M.get_active_directory("/world/commons/robots/")
        robot_xml_root = xmlp.parse(dir + robot_xml).getroot()

        joint_no = 0
        for child in robot_xml_root:
            if child.tag == "bodypart":
                self.body_parts[child.attrib['name']] = Body_Part(child.attrib['mass'])
            elif child.tag == "joint":
                self.joints_info[joint_no] = Joint_Info(child)
                self.joints_position[joint_no] = 0.0
                ji = self.joints_info[joint_no]

                # 如果身体部位是第一个锚点(为了简化模型单向遍历)
                self.body_parts[ji.anchor0_part].joints.append(Robot.MAP_PERCEPTOR_TO_INDEX[ji.perceptor]) 

                joint_no += 1
                if joint_no == self.no_of_joints: break # 忽略额外关节

            else:
                raise NotImplementedError

        assert joint_no == self.no_of_joints, "agentXML和agent类型不匹配！"


    def get_head_abs_vel(self, history_steps:int):
        '''
        获取agent的头部绝对速度(m/s)

        参数
        ----------
        history_steps : int
            考虑的历史步数 [1,40]

        示例
        --------
        get_head_abs_vel(1) 等同于 (当前绝对位置 - 上一个绝对位置)      / 0.04
        get_head_abs_vel(2) 等同于 (当前绝对位置 - 0.08s前的绝对位置) / 0.08
        get_head_abs_vel(3) 等同于 (当前绝对位置 - 0.12s前的绝对位置) / 0.12
        '''
        assert 1 <= history_steps <= 40, "参数 'history_steps' 必须在范围 [1,40]"

        if len(self.loc_head_position_history) == 0:
            return np.zeros(3)

        h_step = min(history_steps, len(self.loc_head_position_history))
        t = h_step * Robot.VISUALSTEP

        return (self.loc_head_position - self.loc_head_position_history[h_step-1]) / t
        

    def _initialize_kinematics(self):

        # 从头部开始
        parts={"head"}
        sequential_body_parts = ["head"]

        while len(parts) > 0:
            part = parts.pop()

            for j in self.body_parts[part].joints:

                p = self.joints_info[j].anchor1_part

                if len(self.body_parts[p].joints) > 0: # 如果身体部位是某个关节的第一个锚点,则添加
                    parts.add(p)
                    sequential_body_parts.append(p)

        self.fwd_kinematics_list = [(self.body_parts[part],j, self.body_parts[self.joints_info[j].anchor1_part] ) 
                                     for part in sequential_body_parts for j in self.body_parts[part].joints]

        # 修复对称性问题 4/4 (运动学)
        for i in Robot.FIX_INDICES_LIST:
            self.joints_info[i].axes *= -1
            aux = self.joints_info[i].min
            self.joints_info[i].min = -self.joints_info[i].max
            self.joints_info[i].max = -aux


    def update_localization(self, localization_raw, time_local_ms): 

        # 解析原始数据
        loc = localization_raw.astype(float) # 32位到64位以保持一致性
        self.loc_is_up_to_date = bool(loc[32])
        self.loc_head_z_is_up_to_date = bool(loc[34])

        if self.loc_head_z_is_up_to_date:
            time_diff = (time_local_ms - self.loc_head_z_last_update) / 1000 
            self.loc_head_z_vel = (loc[33] - self.loc_head_z) / time_diff
            self.loc_head_z = loc[33]
            self.loc_head_z_last_update = time_local_ms

        # 在每个视觉周期保存上一个位置到历史记录(即使不是最新的)(update_localization仅在视觉周期调用)
        self.loc_head_position_history.appendleft(np.copy(self.loc_head_position))

        if self.loc_is_up_to_date:
            time_diff = (time_local_ms - self.loc_last_update) / 1000
            self.loc_last_update = time_local_ms
            self.loc_head_to_field_transform.m[:] = loc[0:16].reshape((4,4))
            self.loc_field_to_head_transform.m[:] = loc[16:32].reshape((4,4))
        
            # 提取数据(与agent头部相关的)
            self.loc_rotation_head_to_field = self.loc_head_to_field_transform.get_rotation()
            self.loc_rotation_field_to_head = self.loc_field_to_head_transform.get_rotation()
            p = self.loc_head_to_field_transform.get_translation()
            self.loc_head_velocity = (p - self.loc_head_position) / time_diff
            self.loc_head_position = p
            self.loc_head_position_last_update = time_local_ms
            self.loc_head_orientation = self.loc_head_to_field_transform.get_yaw_deg()
            self.radio_fallen_state = False

            # 提取数据(与质心相关的)
            p = self.loc_head_to_field_transform(self.rel_cart_CoM_position)
            self.loc_CoM_velocity = (p - self.loc_CoM_position) / time_diff
            self.loc_CoM_position = p

            # 提取数据(与agent躯干相关的)
            t = self.get_body_part_to_field_transform('torso')
            self.loc_torso_to_field_transform = t
            self.loc_torso_to_field_rotation = t.get_rotation()
            self.loc_torso_orientation = t.get_yaw_deg()
            self.loc_torso_pitch = t.get_pitch_deg()
            self.loc_torso_roll = t.get_roll_deg()
            self.loc_torso_inclination = t.get_inclination_deg()
            p = t.get_translation()
            self.loc_torso_velocity = (p - self.loc_torso_position) / time_diff
            self.loc_torso_position = p
            self.loc_torso_acceleration = self.loc_torso_to_field_rotation.multiply(self.acc) + Robot.GRAVITY


    def head_to_body_part_transform(self, body_part_name, coords, is_batch=False):
        '''
        如果coord是一个向量或向量列表：
        将相对于头部的笛卡尔坐标转换为相对于身体部位的坐标

        如果coord是一个Matrix_4x4或Matrix_4x4列表：
        将相对于头部的姿态转换为相对于身体部位的姿态
        
        参数
        ----------
        body_part_name : `str`
            身体部位名称(由agent的XML给出)
        coords : array_like
            一个3D位置或3D位置列表
        is_batch : `bool`
            指示coords是否为3D位置的批处理

        返回
        -------
        coord : `list` 或 ndarray
            如果is_batch为False,则返回一个numpy数组,否则返回一个数组列表
        '''
        head_to_bp_transform : Matrix_4x4 = self.body_parts[body_part_name].transform.invert()
        
        if is_batch:
            return [head_to_bp_transform(c) for c in coords]
        else:
            return head_to_bp_transform(coords)


    def get_body_part_to_field_transform(self, body_part_name) -> Matrix_4x4:
        '''
        计算从身体部位到场的变换矩阵,从中可以提取其绝对位置和旋转。
        为了获得最佳结果,请在self.loc_is_up_to_date为True时使用此方法。否则,前向运动学
        将与定位数据不同步,可能会出现奇怪的结果。
        '''
        return self.loc_head_to_field_transform.multiply(self.body_parts[body_part_name].transform)

    def get_body_part_abs_position(self, body_part_name) -> np.ndarray:
        '''
        计算考虑定位数据和前向运动学的身体部位的绝对位置。
        为了获得最佳结果,请在self.loc_is_up_to_date为True时使用此方法。否则,前向运动学
        将与定位数据不同步,可能会出现奇怪的结果。
        '''
        return self.get_body_part_to_field_transform(body_part_name).get_translation()

    def get_joint_to_field_transform(self, joint_index) -> Matrix_4x4:
        '''
        计算从关节到场的变换矩阵,从中可以提取其绝对位置和旋转。
        为了获得最佳结果,请在self.loc_is_up_to_date为True时使用此方法。否则,前向运动学
        将与定位数据不同步,可能会出现奇怪的结果。
        '''
        return self.loc_head_to_field_transform.multiply(self.joints_transform[joint_index])

    def get_joint_abs_position(self, joint_index) -> np.ndarray:
        '''
        计算考虑定位数据和前向运动学的关节的绝对位置。
        为了获得最佳结果,请在self.loc_is_up_to_date为True时使用此方法。否则,前向运动学
        将与定位数据不同步,可能会出现奇怪的结果。
        '''
        return self.get_joint_to_field_transform(joint_index).get_translation()

    def update_pose(self):

        if self.fwd_kinematics_list is None:
            self._initialize_kinematics()

        for body_part, j, child_body_part in self.fwd_kinematics_list:
            ji = self.joints_info[j]
            self.joints_transform[j].m[:] = body_part.transform.m
            self.joints_transform[j].translate(ji.anchor0_axes, True)
            child_body_part.transform.m[:] = self.joints_transform[j].m
            child_body_part.transform.rotate_deg(ji.axes, self.joints_position[j], True)
            child_body_part.transform.translate(ji.anchor1_axes_neg, True)

        self.rel_cart_CoM_position = np.average([b.transform.get_translation() for b in self.body_parts.values()], 0,
                                                [b.mass                        for b in self.body_parts.values()])


    def update_imu(self, time_local_ms):

        # 更新IMU
        if self.loc_is_up_to_date:
            self.imu_torso_roll = self.loc_torso_roll
            self.imu_torso_pitch = self.loc_torso_pitch   
            self.imu_torso_orientation = self.loc_torso_orientation
            self.imu_torso_inclination = self.loc_torso_inclination
            self.imu_torso_to_field_rotation.m[:] = self.loc_torso_to_field_rotation.m
            self.imu_weak_torso_to_field_transform.m[:] = self.loc_torso_to_field_transform.m
            self.imu_weak_head_to_field_transform.m[:] = self.loc_head_to_field_transform.m
            self.imu_weak_field_to_head_transform.m[:] = self.loc_field_to_head_transform.m
            self.imu_weak_torso_position[:] = self.loc_torso_position
            self.imu_weak_torso_velocity[:] = self.loc_torso_velocity
            self.imu_weak_torso_acceleration[:] = self.loc_torso_acceleration
            self.imu_weak_torso_next_position = self.loc_torso_position + self.loc_torso_velocity * Robot.STEPTIME + self.loc_torso_acceleration * (0.5 * Robot.SQ_STEPTIME)
            self.imu_weak_torso_next_velocity = self.loc_torso_velocity + self.loc_torso_acceleration * Robot.STEPTIME
            self.imu_weak_CoM_position[:] = self.loc_CoM_position
            self.imu_weak_CoM_velocity[:] = self.loc_CoM_velocity
            self.imu_last_visual_update = time_local_ms
        else:
            g = self.gyro / 50 # 将度每秒转换为度每步

            self.imu_torso_to_field_rotation.multiply( Matrix_3x3.from_rotation_deg(g), in_place=True, reverse_order=True)

            self.imu_torso_orientation = self.imu_torso_to_field_rotation.get_yaw_deg()
            self.imu_torso_pitch = self.imu_torso_to_field_rotation.get_pitch_deg()
            self.imu_torso_roll = self.imu_torso_to_field_rotation.get_roll_deg()

            self.imu_torso_inclination = atan(sqrt(tan(self.imu_torso_roll/180*pi)**2+tan(self.imu_torso_pitch/180*pi)**2))*180/pi

            # 更新位置和速度,直到上次视觉更新后0.2秒
            if time_local_ms < self.imu_last_visual_update + 200:
                self.imu_weak_torso_position[:] = self.imu_weak_torso_next_position
                if self.imu_weak_torso_position[2] < 0: self.imu_weak_torso_position[2] = 0 # 限制z坐标为正值
                self.imu_weak_torso_velocity[:] = self.imu_weak_torso_next_velocity * Robot.IMU_DECAY # 稳定性权衡
            else:
                self.imu_weak_torso_velocity *= 0.97 # 在没有视觉更新0.2秒后,位置锁定,速度衰减到零

            # 将适当加速度转换为坐标加速度并修正舍入偏差
            self.imu_weak_torso_acceleration = self.imu_torso_to_field_rotation.multiply(self.acc) + Robot.GRAVITY
            self.imu_weak_torso_to_field_transform = Matrix_4x4.from_3x3_and_translation(self.imu_torso_to_field_rotation,self.imu_weak_torso_position)
            self.imu_weak_head_to_field_transform = self.imu_weak_torso_to_field_transform.multiply(self.body_parts["torso"].transform.invert())
            self.imu_weak_field_to_head_transform = self.imu_weak_head_to_field_transform.invert()
            p = self.imu_weak_head_to_field_transform(self.rel_cart_CoM_position)
            self.imu_weak_CoM_velocity = (p-self.imu_weak_CoM_position)/Robot.STEPTIME
            self.imu_weak_CoM_position = p

            # 下一步位置 = x0 + v0*t + 0.5*a*t^2,   下一步速度 = v0 + a*t
            self.imu_weak_torso_next_position = self.imu_weak_torso_position + self.imu_weak_torso_velocity * Robot.STEPTIME + self.imu_weak_torso_acceleration * (0.5 * Robot.SQ_STEPTIME)
            self.imu_weak_torso_next_velocity = self.imu_weak_torso_velocity + self.imu_weak_torso_acceleration * Robot.STEPTIME



    def set_joints_target_position_direct(self, indices, values:np.ndarray, harmonize=True, max_speed=7.03, tolerance=0.012, limit_joints=True) -> int:
        '''
        计算列表中关节的速度,以达到目标位置

        参数
        ----------
        indices : `int`/`list`/`slice`/numpy array
            关节索引
        values : numpy array 
            每个列出关节索引的目标位置
        harmonize : `bool`
            如果为True,所有关节同时到达目标
        max_speed : `float`
            所有关节的最大速度(deg/step)
            大多数关节的最大速度为351.77 deg/s,根据rcssserver3d/data/rsg/agent/nao/hingejoint.rsg,转换为7.0354 deg/step或6.1395 rad/s
        tolerance : `float`
            角度误差容差(度),如果目标已达到则返回-1
        limit_joints : `bool`
            将值限制在关节的运动范围内

        返回
        -------
        remaining_steps : `int`
            预测的剩余步数,如果目标已达到则返回-1

        示例
        -------
        (设p[tx]为t=x时的关节位置)

        返回值示例：将joint[0]从0度移动到10度
                pos[t0]: 0,  speed[t0]: 7deg/step,  ret=2   # 目标将在2步内达到
                pos[t1]: 7,  speed[t1]: 3deg/step,  ret=1   # 目标将在1步内达到(发送最终动作)
                pos[t2]: 10, speed[t2]: 0deg/step,  ret=0   # 目标已达到
                pos[t3]: 10, speed[t3]: 0deg/step,  ret=-1  # (最佳情况)服务器报告延迟,目标已达到(见容差)
                pos[t?]: 10, speed[t?]: 0deg/step,  ret=-1  # 如果有摩擦,可能需要一些额外步骤

                如果一切按预测进行,我们可以在ret==1时停止调用此函数
                如果需要精度,建议等待ret==-1

        示例1：
            set_joints_target_position_direct(range(2,4),np.array([10.0,5.0]),harmonize=True)    
                Joint[2]   p[t0]: 0  target pos: 10  ->  p[t1]=5,   p[t2]=10
                Joint[3]   p[t0]: 0  target pos: 5   ->  p[t1]=2.5, p[t2]=5

        示例2：
            set_joints_target_position_direct([2,3],np.array([10.0,5.0]),harmonize=False)  
                Joint[2]   p[t0]: 0  target pos: 10  ->  p[t1]=7,   p[t2]=10
                Joint[3]   p[t0]: 0  target pos: 5   ->  p[t1]=5,   p[t2]=5  
        '''

        assert type(values) == np.ndarray, "'values' 参数必须是 numpy 数组"
        np.nan_to_num(values, copy=False) # 将NaN替换为零,将无穷大替换为大有限数

        # 限制关节范围
        if limit_joints:     
            if type(indices) == list or type(indices) == np.ndarray:
                for i in range(len(indices)):
                    values[i] = np.clip(values[i], self.joints_info[indices[i]].min, self.joints_info[indices[i]].max)
            elif type(indices) == slice:
                info = self.joints_info[indices]
                for i in range(len(info)):
                    values[i] = np.clip(values[i], info[i].min, info[i].max)
            else: # int
                values[0] = np.clip(values[0], self.joints_info[indices].min, self.joints_info[indices].max)

        # predicted_diff: 预测的报告位置和实际位置之间的差异

        predicted_diff = self.joints_target_last_speed[indices] * 1.1459156 # rad/s 转换为 deg/step
        predicted_diff = np.asarray(predicted_diff)
        np.clip(predicted_diff,-7.03,7.03,out=predicted_diff) # 就地饱和预测运动

        # reported_dist: 报告位置和目标位置之间的差异

        reported_dist = values - self.joints_position[indices]
        if np.all((np.abs(reported_dist) < tolerance)) and np.all((np.abs(predicted_diff) < tolerance)):
            self.joints_target_speed[indices] = 0
            return -1
       
        deg_per_step = reported_dist - predicted_diff

        relative_max = np.max( np.abs(deg_per_step) ) / max_speed
        remaining_steps = np.ceil( relative_max  )

        if remaining_steps == 0:
            self.joints_target_speed[indices] = 0
            return 0

        if harmonize:   
            deg_per_step /= remaining_steps
        else:
            np.clip(deg_per_step,-max_speed,max_speed,out=deg_per_step) # 限制最大速度

        self.joints_target_speed[indices] = deg_per_step * 0.87266463 # 转换为 rad/s

        return remaining_steps
