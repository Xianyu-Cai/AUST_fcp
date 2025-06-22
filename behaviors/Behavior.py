import numpy as np

class Behavior():

    def __init__(self, base_agent) -> None:
        from agent.Base_Agent import Base_Agent # 用于类型提示
        self.base_agent : Base_Agent = base_agent
        self.world = self.base_agent.world
        self.state_behavior_name = None
        self.state_behavior_init_ms = 0
        self.previous_behavior = None
        self.previous_behavior_duration = None

        # 初始化标准行为
        from behaviors.Poses import Poses
        from behaviors.Slot_Engine import Slot_Engine
        from behaviors.Head import Head

        self.poses = Poses(self.world)
        self.slot_engine = Slot_Engine(self.world)
        self.head = Head(self.world)


    def create_behaviors(self):   
        '''
        行为字典:
            创建:   键: ( 描述, 自动头部, lambda reset[,a,b,c,..]: self.execute(...), lambda: self.is_ready(...) )
            使用:   键: ( 描述, 自动头部, execute_func(reset, *args), is_ready_func )
        '''
        self.behaviors = self.poses.get_behaviors_callbacks()
        self.behaviors.update(self.slot_engine.get_behaviors_callbacks())
        self.behaviors.update(self.get_custom_callbacks())


    def get_custom_callbacks(self):
        '''
        搜索自定义行为可以自动实现
        然而，对于代码分发，动态加载代码并不理想（除非我们加载字节码或其他导入解决方案）
        目前，添加自定义行为是一个手动过程:
            1. 添加导入语句
            2. 将类添加到 'classes' 列表
        '''

        # 行为声明
        from behaviors.custom.Basic_Kick.Basic_Kick import Basic_Kick
        from behaviors.custom.Dribble.Dribble import Dribble
        from behaviors.custom.Fall.Fall import Fall
        from behaviors.custom.Get_Up.Get_Up import Get_Up
        from behaviors.custom.Pass.Pass import Pass
        from behaviors.custom.Step.Step import Step
        from behaviors.custom.Walk.Walk import Walk

        classes = [Basic_Kick,Dribble,Fall,Get_Up,Pass,Step,Walk]

        '''---- 手动声明结束 ----'''

        # 准备回调
        self.objects = {cls.__name__ : cls(self.base_agent) for cls in classes}

        return {name: (o.description,o.auto_head,
                       lambda reset,*args,o=o: o.execute(reset,*args), lambda *args,o=o: o.is_ready(*args)) for name, o in self.objects.items()}


    def get_custom_behavior_object(self, name):
        ''' 获取类 "name" 的唯一对象（"name" 必须代表一个自定义行为） '''
        assert name in self.objects, f"没有名为 {name} 的自定义行为"
        return self.objects[name]
        

    def get_all_behaviors(self):
        ''' 获取所有行为的名称和描述 '''
        return [ key for key in self.behaviors ], [ val[0] for val in self.behaviors.values() ]


    def get_current(self):
        ''' 获取当前行为的名称和持续时间（以秒为单位） '''
        duration = (self.world.time_local_ms - self.state_behavior_init_ms) / 1000.0
        return self.state_behavior_name, duration

    
    def get_previous(self):
        ''' 获取上一个行为的名称和持续时间（以秒为单位） '''
        return self.previous_behavior, self.previous_behavior_duration


    def force_reset(self):
        ''' 强制重置下一个执行的行为 '''
        self.state_behavior_name = None


    def execute(self, name, *args) -> bool:
        ''' 
        执行行为 `name` 的一步，带有参数 `*args`
        - 在第一次调用时自动重置行为
        - 调用 get_current() 获取当前行为（及其持续时间）

        返回
        -------
        finished : bool
            True 如果行为已完成
        '''

        assert name in self.behaviors, f"行为 {name} 不存在!"

        # 检查是否从其他行为转换
        reset = bool(self.state_behavior_name != name)
        if reset: 
            if self.state_behavior_name is not None:
                self.previous_behavior = self.state_behavior_name # 上一个行为被中断（未完成）
            self.previous_behavior_duration = (self.world.time_local_ms - self.state_behavior_init_ms) / 1000.0
            self.state_behavior_name = name
            self.state_behavior_init_ms = self.world.time_local_ms

        # 如果行为允许，控制头部方向
        if self.behaviors[name][1]:
            self.head.execute()

        # 执行行为
        if not self.behaviors[name][2](reset,*args):
            return False

        # 行为已完成
        self.previous_behavior = self.state_behavior_name # 存储当前行为名称
        self.state_behavior_name = None
        return True


    def execute_sub_behavior(self, name, reset, *args):
        '''
        执行行为 `name` 的一步，带有参数 `*args`
        对于调用其他行为的自定义行为很有用
        - 行为重置手动执行
        - 调用 get_current() 将返回主行为（不是子行为）
        - Poses 忽略 reset 参数

        返回
        -------
        finished : bool
            True 如果行为已完成
        '''

        assert name in self.behaviors, f"行为 {name} 不存在!"

        # 如果行为允许，控制头部方向
        if self.behaviors[name][1]:
            self.head.execute()

        # 执行行为
        return self.behaviors[name][2](reset,*args)

    
    def execute_to_completion(self, name, *args):
        ''' 
        执行步骤并与服务器通信直到完成 
        - Slot 行为在发送最后一个命令时（立即发送）表示行为已完成
        - Poses 在服务器返回所需机器人状态时完成（因此最后一个命令无关紧要）
        - 对于自定义行为，我们假设相同的逻辑，因此最后一个命令被忽略

        注意
        -----
        - 退出前，重置 `Robot.joints_target_speed` 数组以避免污染下一个命令
        - 对于在第一次调用时表示已完成行为的 Poses 和自定义行为，不提交或发送任何内容
        - 警告：如果行为永不结束，此函数可能会陷入无限循环
        '''

        r = self.world.robot
        skip_last = name not in self.slot_engine.behaviors

        while True:
            done = self.execute(name, *args)
            if done and skip_last: break # 如果最后一个命令无关紧要，在这里退出
            self.base_agent.scom.commit_and_send( r.get_command() ) 
            self.base_agent.scom.receive()
            if done: break # 如果最后一个命令是行为的一部分，在这里退出

        # 重置以避免污染下一个命令
        r.joints_target_speed = np.zeros_like(r.joints_target_speed)


    def is_ready(self, name, *args) -> bool:
        ''' 检查行为在当前游戏/机器人条件下是否准备好开始 '''

        assert name in self.behaviors, f"行为 {name} 不存在!"
        return self.behaviors[name][3](*args)