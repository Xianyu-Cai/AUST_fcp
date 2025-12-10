from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
from math_ops.Neural_Network import run_mlp
import pickle
import numpy as np
import os

class Wave():
    '''
    挥手行为 - 控制机器人手臂进行挥手动作
    
    使用方法:
        behavior.execute("Wave")  # 执行一步挥手动作
        behavior.execute("Wave", duration=3.0)  # 指定挥手持续时间(秒)
    '''

    def __init__(self, base_agent: Base_Agent) -> None:
        self.base_agent = base_agent
        self.world = base_agent.world
        self.description = "Wave arm gesture (挥手动作)"
        self.auto_head = True  # 允许自动控制头部
        
        # 右手臂关节索引
        self.arm_joints = [
            self.world.robot.J_RARM_PITCH,    # 15 - 手臂前后摆动
            self.world.robot.J_RARM_ROLL,     # 17 - 手臂左右摆动
            self.world.robot.J_RELBOW_YAW,    # 19 - 肘部旋转
            self.world.robot.J_RELBOW_ROLL    # 21 - 肘部弯曲
        ]
        
        # 状态变量
        self.step_counter = 0
        self.wave_duration = 3.0  # 默认挥手持续时间(秒)
        self.start_time = 0
        
        # 尝试加载训练好的模型 (如果存在)
        self.model = None
        model_path = M.get_active_directory("/behaviors/custom/Wave/wave.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Wave: 已加载训练模型")
            except Exception as e:
                print(f"Wave: 加载模型失败 - {e}")
        
        # 观察空间大小 (用于神经网络)
        self.obs = np.zeros(9, dtype=np.float32)

    def observe(self):
        '''获取当前观察状态'''
        r = self.world.robot
        
        # 手臂关节位置 (归一化)
        for i, joint_idx in enumerate(self.arm_joints):
            self.obs[i] = r.joints_position[joint_idx] / 100.0
        
        # 时间步 (归一化)
        elapsed_time = (self.world.time_local_ms - self.start_time) / 1000.0
        self.obs[4] = min(elapsed_time / self.wave_duration, 1.0)
        
        # 头部高度
        self.obs[5] = r.loc_head_z
        
        # 陀螺仪数据 (归一化)
        self.obs[6:9] = r.gyro / 100.0
        
        return self.obs

    def execute(self, reset, duration=3.0) -> bool:
        '''
        执行挥手动作
        
        Parameters
        ----------
        reset : bool
            是否重置行为状态
        duration : float
            挥手持续时间(秒), 默认3.0秒
            
        Returns
        -------
        finished : bool
            True 如果挥手动作完成
        '''
        r = self.world.robot
        
        if reset:
            self.step_counter = 0
            self.start_time = self.world.time_local_ms
            self.wave_duration = duration
        
        self.step_counter += 1
        elapsed_time = (self.world.time_local_ms - self.start_time) / 1000.0
        
        # 检查是否完成
        if elapsed_time >= self.wave_duration:
            # 恢复手臂到默认位置
            self._set_default_arm_pose(r)
            return True
        
        # 根据是否有训练模型选择执行方式
        if self.model is not None:
            self._execute_with_model(r)
        else:
            self._execute_scripted(r, elapsed_time)
        
        return False

    def _execute_with_model(self, r):
        '''使用训练好的神经网络模型执行动作'''
        obs = self.observe()
        action = run_mlp(obs, self.model)
        
        # 缩放动作
        scaled_action = action * 15
        
        # 基础挥手姿势
        base_pose = np.array([-60, -30, 0, 45], dtype=np.float32)
        
        # 添加周期性挥手动作
        wave_offset = np.sin(self.step_counter * 0.3) * 20
        base_pose[1] += wave_offset
        
        # 应用学习的调整
        target_angles = base_pose + scaled_action
        
        r.set_joints_target_position_direct(
            self.arm_joints,
            target_angles,
            harmonize=False
        )

    def _execute_scripted(self, r, elapsed_time):
        '''使用脚本化的挥手动作 (无模型时使用)'''
        # 挥手频率
        wave_freq = 3.0  # Hz
        
        # 基础姿势: 手臂抬起
        arm_pitch = -70   # 手臂向上抬起
        arm_roll_base = -40  # 手臂向外
        elbow_yaw = 0     # 肘部不旋转
        elbow_roll = 50   # 肘部弯曲
        
        # 挥手动作: 在roll方向上左右摆动
        wave_amplitude = 30
        arm_roll = arm_roll_base + wave_amplitude * np.sin(2 * np.pi * wave_freq * elapsed_time)
        
        # 渐入渐出
        fade_time = 0.5  # 过渡时间
        if elapsed_time < fade_time:
            # 渐入
            factor = elapsed_time / fade_time
            arm_pitch = -90 + (arm_pitch + 90) * factor  # 从默认位置渐变
            arm_roll = -10 + (arm_roll + 10) * factor
            elbow_roll = 0 + elbow_roll * factor
        elif elapsed_time > self.wave_duration - fade_time:
            # 渐出
            factor = (self.wave_duration - elapsed_time) / fade_time
            arm_pitch = -90 + (arm_pitch + 90) * factor
            arm_roll = -10 + (arm_roll + 10) * factor
            elbow_roll = 0 + elbow_roll * factor
        
        # 设置关节目标
        target_angles = np.array([arm_pitch, arm_roll, elbow_yaw, elbow_roll], dtype=np.float32)
        
        r.set_joints_target_position_direct(
            self.arm_joints,
            target_angles,
            harmonize=True
        )

    def _set_default_arm_pose(self, r):
        '''恢复手臂到默认位置'''
        default_pose = np.array([-90, -10, 0, 0], dtype=np.float32)
        r.set_joints_target_position_direct(
            self.arm_joints,
            default_pose,
            harmonize=True
        )

    def is_ready(self) -> bool:
        '''
        检查行为是否可以开始执行
        
        Returns
        -------
        ready : bool
            True 如果机器人处于可以挥手的状态
        '''
        r = self.world.robot
        # 机器人需要站立 (头部高度 > 0.3m)
        return r.loc_head_z > 0.3
