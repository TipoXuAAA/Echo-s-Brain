from Dependencies import *
from Nervous_system import *
# ==========================================
# 0. 环境适配器
# ==========================================
class MyoAdapter:
    def __init__(self, env_name):
        self.env = myo_gym.make(env_name)
        self.env.reset()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            truncated = False 
            return obs, reward, done, truncated, info
        else:
            return step_result 

    def render_live(self):
        try:
            self.env.mj_render()
        except Exception:
            pass 

    def render_frame(self, flip_vertical=False):
        # 🎥 离屏渲染用于通过视频保存
        rgb = self.env.sim.renderer.render_offscreen(width=640, height=480, camera_id=-1)
        if flip_vertical:
            return np.flipud(rgb)
        return rgb

    def close(self):
        self.env.close()
# ==========================================
# 1. 工具函数：GAE 计算
# ==========================================
def compute_gae(rewards, values, mask, next_value, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (Generalized Advantage Estimation)
    """
    returns = []
    gae = 0
    # 将 next_value 加入 values 列表末尾方便计算
    values = values + [next_value] 
    
    for i in reversed(range(len(rewards))):
        # TD Error (delta) = r + gamma * V(s') * mask - V(s)
        delta = rewards[i] + gamma * values[i + 1] * mask[i] - values[i]
        # GAE = delta + gamma * lambda * mask * GAE(prev)
        gae = delta + gamma * lam * mask[i] * gae
        # Return = GAE + V(s)
        returns.insert(0, gae + values[i])
    
    return returns
# ==========================================
# 2. SurvivalEngine
# ==========================================
class SurvivalEngine:
    """
    生存引擎：负责与环境交互、数据采集、评估表现
    类比动物的「身体」和「世界交互接口」，不包含学习逻辑
    """
    def __init__(self, env_name, brain, run_name="BioBrain_Survival"):
        self.brain = brain  # 注入 BioBrain 实例
        self.device = next(brain.parameters()).device
        
        # 环境与数据缓冲
        self.env = MyoAdapter(env_name)
        self.obs = self.env.reset()
        self.rnn_hidden = None
        
        # 记录与存储
        self.run_name = run_name
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.ckpt_dir = f"checkpoints/{run_name}"
        self.video_dir = f"videos/{run_name}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.global_step = 0
        self.best_reward = -float('inf')
        self.load_checkpoint()

    # === 环境交互核心 ===
    def collect_experience(self, batch_size=4096):
        """采集一批经验（对应Agent的一次「探索过程记录」）"""
        buffers = {
            'obs': [], 'raw_intent': [], 'logp': [], 
            'value': [], 'reward': [], 'mask': [], 'hidden': [],
            'next_obs': [], 
        }
        
        steps = 0
        while steps < batch_size:
            # 1. 决策：获取动作
            obs_t = torch.FloatTensor(self.obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # ✅ 适配 forward 返回值：新增 sensory_state（但当前不需要，用 _ 占位）
                act, logp, val, entropy, next_hidden, raw_intent, sensory_state = self.brain(
                    obs_t, hidden=self.rnn_hidden
                )

            # 2. 执行动作
            next_obs, reward, done, trunc, _ = self.env.step(act.cpu().numpy().flatten())
            # 处理 done 和 trunc 的情况，这些情况需要重置环境和网络隐藏状态。但是后面我们会区分done和trunc两种情况的价值计算
            is_terminal = done or trunc
            
            # 3. 存储经验
            buffers['obs'].append(self.obs)
            buffers['raw_intent'].append(raw_intent.cpu().numpy())
            buffers['logp'].append(logp.item())
            buffers['value'].append(val.item())
            buffers['reward'].append(reward)
            # 如果是done的情况，则要停止加上下一个状态的价值，所以需要个mask来判断
            buffers['mask'].append(0.0 if done else 1.0)
            # 因为保存了next_obs，所以如果是truncated情况，也能像其他正常情况把下一个状态价值加上
            buffers['next_obs'].append(next_obs)
            # 存储当前的 hidden 用于之后梯度计算时的初始状态
            current_h = self.rnn_hidden if self.rnn_hidden is not None else torch.zeros(1, 1, self.brain.config.sensory_latent).to(self.device)
            buffers['hidden'].append(current_h) 
            
            # 4. 状态更新
            self.obs = next_obs if not is_terminal else self.env.reset()
            self.rnn_hidden = next_hidden if not is_terminal else None
            self.global_step += 1
            steps += 1

        # 5. 处理经验（计算 GAE 和 returns）
        return self._process_buffers(buffers)

    def _process_buffers(self, buffers):
        """将原始经验转换为训练用的 batch（含 GAE 计算）"""
        
        # 1. 计算最后一个状态的价值 (Bootstrap Value)
        # 只有在 collect_experience 完成后才进行
        with torch.no_grad():
            last_obs = torch.FloatTensor(buffers['obs'][-1]).unsqueeze(0).to(self.device)
            
            # 这里的逻辑是：如果本轮结束时没有 done，我们就用最后的 rnn_hidden；如果 done 了，理论上价值是0，这里用 None 或全0向量都行
            last_hidden = self.rnn_hidden 
            
            _, _, next_val, _, _, _, _ = self.brain(last_obs, hidden=last_hidden)
            next_val = next_val.item()
        
        # 2. 调用工具函数计算 GAE
        returns = compute_gae(buffers['reward'], buffers['value'], buffers['mask'], next_val)
        
        # 3. 处理 Hidden States 拼接
        hidden_batch = torch.cat(buffers['hidden'], dim=1).to(self.device)

        # 4. 组装 Batch
        batch = {
            'obs': torch.FloatTensor(np.array(buffers['obs'])).to(self.device),
            'raw_intent': torch.FloatTensor(np.array(buffers['raw_intent'])).to(self.device),
            'logp': torch.FloatTensor(np.array(buffers['logp'])).to(self.device),
            'value': torch.FloatTensor(np.array(buffers['value'])).to(self.device),
            'return': torch.FloatTensor(np.array(returns)).to(self.device),
            'advantage': (torch.FloatTensor(np.array(returns)) - torch.FloatTensor(np.array(buffers['value']))).to(self.device),
            'hidden': hidden_batch,
            'next_obs': torch.FloatTensor(np.array(buffers['next_obs'])).to(self.device),
            'mask': torch.FloatTensor(np.array(buffers['mask'])).unsqueeze(1).to(self.device)
        }
        
        # Advantage 标准化（让训练更稳定），用于actor-critic架构下的actor部分，所以不会影响critic的绝对价值学习
        batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
        
        return batch

    # === 评估与记录 ===
    def evaluate_performance(self):
        """评估当前策略表现（无探索）"""
        obs = self.env.reset()
        hidden = None
        total_reward = 0.0
        
        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, hidden = self.brain.get_action_deterministic(obs_t, hidden)
            obs, r, done, trunc, _ = self.env.step(action)
            total_reward += r
            if done or trunc:
                break
        
        return total_reward

    def save_video(self, max_steps=400):
        """录制当前策略的视频（无探索）"""
        video_path = f"{self.video_dir}/step_{self.global_step}.mp4"
        frames = []
        obs = self.env.reset()
        hidden = None
        
        for _ in range(max_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, hidden = self.brain.get_action_deterministic(obs_t, hidden)
            obs, _, done, trunc, _ = self.env.step(action)
            frames.append(self.env.render_frame(flip_vertical=False))
            if done or trunc:
                break
        
        if frames:
            imageio.mimsave(video_path, frames, fps=30)
            print(f"📹 视频保存: {video_path}")

    # ===  checkpoint 管理 ===
    def save_checkpoint(self, score):
        is_best = score > self.best_reward
        if is_best:
            self.best_reward = float(score)

        payload = {
            "brain_state": self.brain.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.brain.optimizers.items()},
            "step": int(self.global_step),
            "best_reward": float(self.best_reward),
        }
        torch.save(payload, f"{self.ckpt_dir}/{'best' if is_best else 'latest'}_checkpoint.pth")

        if is_best:
            print(f"🏆 新纪录: {score:.2f} (已保存)")

    def load_checkpoint(self):
        ckpt_path = glob.glob(f"{self.ckpt_dir}/*.pth")
        if not ckpt_path:
            print("⚠️ 无检查点可加载")
            return

        latest_ckpt = max(ckpt_path, key=os.path.getctime)
        ckpt = torch.load(latest_ckpt, map_location=self.device, weights_only=False)

        self.brain.load_state_dict(ckpt["brain_state"], strict=False)

        saved_opts = ckpt.get("optimizers", {})
        for name, opt in self.brain.optimizers.items():
            state = saved_opts.get(name, None)
            if state is None:
                print(f"⚠️ optimizer[{name}] not found in ckpt, skip.")
                continue
            try:
                opt.load_state_dict(state)
            except ValueError as e:
                print(f"⚠️ optimizer[{name}] state mismatch, skip. ({e})")

        self.global_step = int(ckpt.get("step", 0))
        self.best_reward = float(ckpt.get("best_reward", -float("inf")))
        print(f"🔄 恢复检查点: {latest_ckpt} (step {self.global_step})")