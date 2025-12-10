from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from time import sleep
from world.commons.Draw import Draw
import gym
import numpy as np
import os

'''
Objective:
Learn how to wave (挥手动作)
Train the robot to raise and wave its arm in a greeting gesture
----------
- class Wave: implements an OpenAI custom gym
- class Train: implements algorithms to train a new model or test an existing model
'''

class Wave(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw, [])
        
        # 挥手动作相关的关节索引 (右手臂)
        # J_RARM_PITCH=15, J_RARM_ROLL=17, J_RELBOW_YAW=19, J_RELBOW_ROLL=21
        self.arm_joints = [15, 17, 19, 21]  # 右手臂关节
        
        # 动作空间: 4个手臂关节的目标角度变化 (归一化到 [-1, 1])
        self.action_space = gym.spaces.Box(
            low=np.full(4, -1.0, np.float32), 
            high=np.full(4, 1.0, np.float32), 
            dtype=np.float32
        )
        
        # 观察空间: 关节位置(4) + 时间步(1) + 头部高度(1) + 陀螺仪(3) = 9
        # 使用合理的边界值
        obs_low = np.array([-1.8, -1.8, -1.8, -1.8, 0.0, 0.0, -5.0, -5.0, -5.0], dtype=np.float32)
        obs_high = np.array([1.8, 1.8, 1.8, 1.8, 1.0, 1.0, 5.0, 5.0, 5.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, 
            high=obs_high, 
            dtype=np.float32
        )
        
        self.step_counter = 0
        self.max_steps = 100  # 每个episode的最大步数
        self.wave_phase = 0  # 挥手阶段

    def reset(self):
        # 传送到初始位置
        self.player.scom.commit_beam((-3, 0), 0)
        
        # 等待机器人稳定
        for _ in range(20):
            self.player.behavior.execute("Zero_Bent_Knees")
            r = self.player.world.robot
            self.player.scom.commit_and_send(r.get_command())
            self.player.scom.receive()
        
        self.step_counter = 0
        self.wave_phase = 0
        
        return self._get_obs()

    def _get_obs(self):
        r = self.player.world.robot
        obs = np.zeros(9, dtype=np.float32)
        
        # 手臂关节位置 (归一化)
        for i, joint_idx in enumerate(self.arm_joints):
            obs[i] = r.joints_position[joint_idx] / 100.0
        
        # 时间步 (归一化)
        obs[4] = self.step_counter / self.max_steps
        
        # 头部高度
        obs[5] = r.loc_head_z
        
        # 陀螺仪数据 (归一化)
        obs[6:9] = r.gyro / 100.0
        
        return obs

    def step(self, action):
        r = self.player.world.robot
        
        # 缩放动作 (将神经网络输出映射到实际角度范围)
        scaled_action = action * 15  # 缩放因子
        
        # 目标手臂姿势 (基础挥手姿势 + 学习的调整)
        # 基础姿势: 手臂抬起并弯曲
        base_pose = np.array([-60, -30, 0, 45], dtype=np.float32)  # pitch, roll, elbow_yaw, elbow_roll
        
        # 添加周期性挥手动作
        wave_offset = np.sin(self.step_counter * 0.3) * 20  # 挥手摆动
        base_pose[1] += wave_offset  # 在roll方向上摆动
        
        # 应用学习的动作调整
        target_angles = base_pose + scaled_action
        
        # 设置手臂关节目标位置
        r.set_joints_target_position_direct(
            self.arm_joints,
            target_angles,
            harmonize=False
        )
        
        # 保持站立姿势 (腿部使用默认姿势)
        self.player.behavior.execute("Zero_Bent_Knees")
        
        # 发送命令并接收状态
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()
        
        self.step_counter += 1
        
        # 计算奖励
        reward = self._compute_reward(r)
        
        # 检查是否结束
        done = self.step_counter >= self.max_steps or r.loc_head_z < 0.3
        
        return self._get_obs(), reward, done, {}

    def _compute_reward(self, r):
        reward = 0.0
        
        # 1. 保持站立奖励 (头部高度越高越好)
        reward += r.loc_head_z * 0.5
        
        # 2. 手臂抬起奖励 (pitch角度越负表示手臂越高)
        arm_pitch = r.joints_position[15]  # J_RARM_PITCH
        if arm_pitch < -30:  # 手臂抬起
            reward += 0.3
        
        # 3. 稳定性奖励 (惩罚过大的身体晃动)
        gyro_penalty = np.sum(np.abs(r.gyro)) / 100.0
        reward -= gyro_penalty * 0.1
        
        # 4. 挥手动作奖励 (手臂在摆动)
        # 使用关节目标速度作为运动指标
        arm_roll_velocity = abs(r.joints_target_speed[17]) if r.joints_target_speed is not None else 0
        reward += min(arm_roll_velocity * 0.01, 0.2)
        
        return reward

    def render(self, mode='human', close=False):
        return

    def close(self):
        Draw.clear_all()
        self.player.scom.close()


class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        n_envs = min(8, os.cpu_count())
        n_steps_per_env = 256
        minibatch_size = 64
        total_steps = 100000
        learning_rate = 3e-4
        folder_name = f'Wave_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'

        print("Model path:", model_path)

        def init_env(i_env):
            def thunk():
                return Wave(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env, self.robot_type, False)
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)

        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:
                model = PPO.load(args["model_file"], env=env, n_envs=n_envs, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate)
            else:
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env, batch_size=minibatch_size, learning_rate=learning_rate)

            model_path = self.learn_model(model, total_steps, model_path, eval_env=eval_env, eval_freq=n_steps_per_env * 10, backup_env_file=__file__)
        except KeyboardInterrupt:
            sleep(1)
            print("\nctrl+c pressed, aborting...\n")
            servers.kill()
            return

        env.close()
        eval_env.close()
        servers.kill()

    def test(self, args):
        server = Server(self.server_p - 1, self.monitor_p, 1)
        env = Wave(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)

        self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])

        env.close()
        server.kill()
