from agent.Base_Agent import Base_Agent as Agent
from behaviors.custom.Dribble.Env import Env as DribbleEnv
from scripts.commons.Server import Server
from scripts.commons.Train_Base import Train_Base
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from time import sleep
import gym
import numpy as np
import os

class Dribble(gym.Env):
    def __init__(self, ip, server_p, monitor_p, r_type, enable_draw) -> None:
        self.robot_type = r_type
        self.player = Agent(ip, server_p, monitor_p, 1, self.robot_type, "Gym", True, enable_draw)
        width = 0.9 if self.robot_type == 3 else 1.2
        self.env = DribbleEnv(self.player, width)

        obs_size = len(self.env.obs)
        self.obs = np.zeros(obs_size, np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(obs_size, -np.inf, np.float32),
                                                high=np.full(obs_size, np.inf, np.float32),
                                                dtype=np.float32)

        act_size = 16
        MAX = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(low=np.full(act_size, -MAX, np.float32),
                                           high=np.full(act_size, MAX, np.float32),
                                           dtype=np.float32)

    def sync(self):
        r = self.player.world.robot
        self.player.scom.commit_and_send(r.get_command())
        self.player.scom.receive()

    def observe(self, init=False):
        return self.env.observe(init, True)

    def reset(self):
        r = self.player.world.robot
        self.player.scom.unofficial_beam((-3, 0, r.beam_height), 0)
        self.player.scom.unofficial_move_ball((-2, 0, 0.042))
        self.sync()
        self.obs[:] = self.observe(True)
        self.lastx = self.player.world.ball_cheat_abs_pos[0]
        return self.obs

    def step(self, action):
        w = self.player.world
        self.env.execute(action)
        self.sync()
        reward = w.ball_cheat_abs_pos[0] - self.lastx
        self.lastx = w.ball_cheat_abs_pos[0]
        done = w.robot.cheat_abs_pos[2] < 0.3 or self.env.step_counter > 200
        return self.observe(), reward, done, {}

    def render(self, mode="human", close=False):
        return

    def close(self):
        self.player.terminate()

class Train(Train_Base):
    def __init__(self, script) -> None:
        super().__init__(script)

    def train(self, args):
        n_envs = min(8, os.cpu_count())
        n_steps_per_env = 256
        minibatch_size = 64
        total_steps = 5000000
        learning_rate = 3e-4
        folder_name = f'Dribble_R{self.robot_type}'
        model_path = f'./scripts/gyms/logs/{folder_name}/'
        print("Model path:", model_path)

        def init_env(i_env):
            def thunk():
                return Dribble(self.ip, self.server_p + i_env, self.monitor_p_1000 + i_env,
                               self.robot_type, False)
            return thunk

        servers = Server(self.server_p, self.monitor_p_1000, n_envs + 1)
        env = SubprocVecEnv([init_env(i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([init_env(n_envs)])

        try:
            if "model_file" in args:
                model = PPO.load(args["model_file"], env=env, n_envs=n_envs,
                                 n_steps=n_steps_per_env, batch_size=minibatch_size,
                                 learning_rate=learning_rate)
            else:
                model = PPO("MlpPolicy", env=env, verbose=1, n_steps=n_steps_per_env,
                             batch_size=minibatch_size, learning_rate=learning_rate)
            self.learn_model(model, total_steps, model_path, eval_env=eval_env,
                             eval_freq=n_steps_per_env*20, backup_env_file=__file__)
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
        env = Dribble(self.ip, self.server_p - 1, self.monitor_p, self.robot_type, True)
        model = PPO.load(args["model_file"], env=env)

        try:
            self.export_model(args["model_file"], args["model_file"]+".pkl", False)
            self.test_model(model, env, log_path=args["folder_dir"], model_path=args["folder_dir"])
        except KeyboardInterrupt:
            print()

        env.close()
        server.kill()
