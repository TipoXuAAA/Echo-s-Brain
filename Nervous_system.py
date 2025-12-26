from Dependencies import *
# ==========================================
# 0. 工具类网络 (Auxilary Networks)
# ==========================================
class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.zeros(())) # 这里的()表示标量
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def forward(self, x):
        if self.training:
            self.update(x)
        return (x - self.mean) / (torch.sqrt(self.var + self.epsilon))
# ==========================================
# 1. 类脑神经网络 (Bio-Inspired Architecture)
# ==========================================
class BrainConfig:
    def __init__(self, n_muscles, obs_dim):
        self.n_muscles = n_muscles    
        self.obs_dim = obs_dim        
        
        # --- 维度定义 (参考你的图) ---
        self.sensory_latent = 256     # S1/V1 输出特征维度
        self.intent_dim = 32          # M1 输出的"运动意图"维度 (低维指令)
        self.synergy_dim = 64         # 脑干红核的"协同模式"数量
        self.hidden_dim = 256         # 中间层神经元数量
# ==========================================
# 1. 感觉皮层 (Sensory Cortex, S1/V1)
# 对应: Observation Encoder, GRU Memory
# ==========================================
class SensoryCortex(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 自适应输入变化幅度
        self.normalizer = RunningMeanStd(config.obs_dim)
        # 模拟 S1 体感皮层和 V1 视觉皮层的整合
        self.encoder = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), # LayerNorm 模拟神经元群体稳态
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.sensory_latent),
            nn.LeakyReLU(0.1)
        )
        # 增加短时记忆 (类似前额叶/海马体交互)
        self.memory_rnn = nn.GRU(config.sensory_latent, config.sensory_latent, batch_first=True)

    def forward(self, obs, hidden=None):
        # 0. 进行归一化处理
        norm_obs = self.normalizer(obs)

        # 1. 编码感知
        x = self.encoder(norm_obs)
        
        # 2. 记忆处理
        perceptual_state, next_hidden = self.memory_rnn(x.unsqueeze(1), hidden)  # 强制 [B, 1, 256]
        perceptual_state = perceptual_state.squeeze(1)  # 输出 [B, 256]
        
        return perceptual_state, next_hidden
# ==========================================
# 2. 运动皮层 (Motor Cortex, M1/Premotor) & 基底核 (Basal Ganglia)
# 对应: Joint Target Generator, Value Estimator
# ==========================================
class MotorCortex(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- Value Stream (基底核/VTA - 评估状态好坏) ---
        self.critic = nn.Sequential(
            nn.Linear(config.sensory_latent, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # --- Policy Stream (M1 - 生成高层运动意图) ---
        # 注意：这里不直接输出肌肉，而是输出"意图"(Intent)，目的是在低秩空间进行学习，减小学习复杂度，然后脑干负责翻译成高维肌肉指令。
        self.actor_trunk = nn.Sequential(
            nn.Linear(config.sensory_latent, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # 输出均值 (Mean Intent)
        self.fc_mean = nn.Linear(config.hidden_dim, config.intent_dim)
        
        # 输出方差 (Log Std) - 🌟 随状态变化！
        # 这允许大脑在不确定时增加探索(高std)，熟练时精确控制(低std)
        self.fc_logstd = nn.Linear(config.hidden_dim, config.intent_dim)

    def forward(self, sensory_state):
        # 价值评估
        value = self.critic(sensory_state)
        
        # 意图分布参数
        x = self.actor_trunk(sensory_state)
        mu = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        
        # 限制 log_std 范围防止数值不稳定 (模拟生物神经元放电噪声的物理极限)
        log_std = torch.clamp(log_std, -20, 2) 
        std = torch.exp(log_std)
        
        return mu, std, value
# ==========================================
# 3. 小脑 (Cerebellum) - 修正与协调
# 对应: World Model / Coordination
# ==========================================
class Cerebellum(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 输入维度 = 意图维度(32) + 感觉状态维度(256)
        input_dim = config.intent_dim + config.sensory_latent
        hidden_dim = config.hidden_dim  # 256
        
        # === 校正网络 (输出意图修正量) ===
        self.corrector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 稳定训练
            nn.Tanh(),  # 输出范围[-1,1]，便于控制修正强度
            nn.Linear(hidden_dim, config.intent_dim)  # 输出32维修正量
        )
        
        # === 状态预测网络 (输出下一时刻感觉状态) ===
        self.state_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),  # 加宽中间层提升预测能力
            nn.LeakyReLU(0.1),  # 避免神经元死亡
            nn.Linear(hidden_dim * 2, config.sensory_latent)  # 输出256维预测
        )
    def predict_next_state(self, intent, sensory_state):
        """
        世界模型：预测 s_{t+1}，假定 batch 内样本相互独立
        Args:
            intent:        [B, intent_dim]
            sensory_state: [B, sensory_latent]
        Returns:
            pred_next_sensory: [B, sensory_latent]
        """

        combined = torch.cat([intent, sensory_state], dim=-1)
        return self.state_predictor(combined)
    def compute_correction(self, intent, sensory_state):
        """
        计算运动意图的微小修正量
        """
        assert intent.dim() == 2
        assert sensory_state.dim() == 2

        combined = torch.cat([intent, sensory_state], dim=-1)
        return self.corrector(combined)
    
    def forward(self, original_intent, sensory_state):
        """
        Args:
            original_intent: [Batch, 32] 或 [Batch, Seq, 32] 原始意图
            sensory_state:   [Batch, 256] 或 [Batch, Seq, 256] 当前感觉状态
        Returns:
            refined_intent: [Batch, 32] 或 [Batch, Seq, 32] 修正后意图
            pred_next_sensory: [Batch, 256] 预测的下一时刻感觉状态
        """
        # === ✅ 工程级止血：统一拉平成 2D ===
        if original_intent.dim() == 3:
            # [B, 1, 32] -> [B, 32]
            original_intent = original_intent.squeeze(1)

        if sensory_state.dim() == 3:
            # [B, 1, 256] -> [B, 256]
            sensory_state = sensory_state.squeeze(1)

        # 直接拼接 [B, 32] + [B, 256] = [B, 288]
        combined = torch.cat([original_intent, sensory_state], dim=-1)
        
        # 计算修正量 [B, 32]
        correction = self.corrector(combined)
        
        # 预测下一状态 [B, 256]
        pred_next_sensory = self.state_predictor(combined)
        
        # 修正意图（直接加，无需 squeeze）
        refined_intent = original_intent + 0.1*correction
        
        return refined_intent, pred_next_sensory

# ==========================================
# 4. 脑干 (Brainstem) - 肌肉协同 (Muscle Synergy)
# 对应: Muscle Synergy Layer (Red Nucleus)
# ==========================================
class Brainstem(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 将"意图" (32维) 解码为 "肌肉协同" (64维) 再映射到 "具体肌肉" (80维)
        # 这就是传说中的"降维打击"的逆过程
        self.synergy_matrix = nn.Sequential(
            nn.Linear(config.intent_dim, config.synergy_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(config.synergy_dim, config.n_muscles)
        )
        
        # 同时也负责调节反射增益 (即调节脊髓的 Gamme 运动神经元)
        self.gain_controller = nn.Linear(config.intent_dim, config.n_muscles)

        # 是否有意义的运动网络
        self.action_to_sensory = nn.Sequential(
            nn.Linear(config.n_muscles, config.sensory_latent),
            nn.LayerNorm(config.sensory_latent),
            nn.Tanh()
        )

    def forward(self, corrected_intent):
        base_muscle_forces = self.synergy_matrix(corrected_intent)
        reflex_gain = torch.tanh(self.gain_controller(corrected_intent))
        return base_muscle_forces, reflex_gain
# ==========================================
# 5. 脊髓 (Spinal Cord) - 反射层
# 对应: Spinal Reflex Layer
# ==========================================
class SpinalCord(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_muscles = config.n_muscles
        
        # 🌟 新增：本体感觉网络需要结合意图（输入=原始obs+意图）
        self.proprioception_net = nn.Sequential(
            nn.Linear(config.obs_dim + config.intent_dim, 128),  # 输入维度 += intent_dim
            nn.ReLU(),
            nn.Linear(128, config.n_muscles * 2) 
        )

        # 动态调整刚度参数,学好后固定住，接下来brainstem会修改刚度Kp
        self.kp_net = nn.Sequential(
            nn.Linear(config.obs_dim + config.intent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.n_muscles) 
        )
        self.kd_net = nn.Sequential(
            nn.Linear(config.obs_dim + config.intent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.n_muscles) 
        )

        
    
    def forward(self, raw_obs, brain_command, gain_modulation, intent):  # 🌟 新增intent参数
        """
        Args:
            intent: [Batch, intent_dim] 皮层下发的运动意图（用于调整本体感觉）
        """
        # 🌟 拼接原始obs和意图，再提取本体感觉
        obs_with_intent = torch.cat([raw_obs, intent], dim=-1)
        proprio = self.proprioception_net(obs_with_intent)
        m_len, m_vel = proprio[:, :self.n_muscles], proprio[:, self.n_muscles:]
        # 动态调整刚度
        kp = self.kp_net(obs_with_intent) * (1.0 + gain_modulation)
        kd = self.kd_net(obs_with_intent)

        # 牵张反射
        reflex = -kp * m_len - kd * m_vel 
        
        return torch.sigmoid(brain_command + reflex)
# ==========================================
# 🌟 最终整合：生物类脑控制核心 (BioBrain)
# ==========================================
class BioBrain(nn.Module):
    def __init__(self, n_muscles, obs_dim):
        super().__init__()
        self.config = BrainConfig(n_muscles, obs_dim)
        
        # 按解剖学层级实例化
        self.sensory_cortex = SensoryCortex(self.config) # Sensory Cortex: 感知
        self.motor_cortex = MotorCortex(self.config)     # MO+ACB: 决策
        self.cerebellum = Cerebellum(self.config)        # 小脑: 预测与微调
        self.brainstem = Brainstem(self.config)          # 脑干: 协同降维
        self.spinal_cord = SpinalCord(self.config)       # 脊髓: 反射执行
        
        self.apply(self._init_weights)

        # 定义优化器组
        self.optimizers = {
            'cortex': optim.Adam(
                list(self.sensory_cortex.parameters()) +
                list(self.motor_cortex.parameters()),
                lr=3e-4
            ),

            'cerebellum_pred': optim.Adam(
                self.cerebellum.state_predictor.parameters(),
                lr=1e-3
            ),

            'cerebellum_corr': optim.Adam(
                self.cerebellum.corrector.parameters(),
                lr=3e-4
            ),

            'brainstem_spinal': optim.Adam(
                list(self.brainstem.parameters()) +
                list(self.spinal_cord.parameters()),
                lr=5e-4
            )
        }

        self.losses = {}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, obs, hidden=None, action_taken=None):
        """
        类脑前向：感觉 → 意图 →（小脑修正）→ 脑干 → 脊髓
        """

        # 感觉皮层：我现在处于什么状态？
        sensory_state, next_hidden = self.sensory_cortex(obs, hidden)

        # 运动皮层：基于感觉生成“意图分布”
        intent_mu, intent_std, value = self.motor_cortex(sensory_state)
        dist = torch.distributions.Normal(intent_mu, intent_std)

        if action_taken is None:
            raw_intent = dist.sample()          # 采样意图（训练/探索）
        else:
            # PPO 更新阶段复用旧意图
            # ===============================================================
            # 🛡️ Echo 的维度卫士 (Shape Guard)
            # 防止 [4096, 1, 6] vs [4096, 6] 导致的 4096*4096 维爆炸
            # ===============================================================
            raw_intent = action_taken
            
            # 1. 如果维度多了 (比如 [B, 1, Dim])，就把中间那个 squeeze 掉
            if raw_intent.dim() > intent_mu.dim():
                raw_intent = raw_intent.squeeze() 
                
            # 2. 双重保险：强制 reshape 成和 intent_mu 一样的形状
            # 只要元素总数一样，这一步能救命
            if raw_intent.shape != intent_mu.shape:
                raw_intent = raw_intent.view_as(intent_mu)

        # 小脑：预测 + 微调意图（不直接执行）
        refined_intent, pred_next_sensory = self.cerebellum(raw_intent, sensory_state)

        # ✅ 关键边界：执行用意图，禁止反向塑造皮层
        refined_intent_exec = refined_intent.detach()

        # 脑干：意图 → 协同肌群信号
        base_forces, reflex_gain = self.brainstem(refined_intent_exec)

        # 脊髓：感觉反射 + 执行动作
        final_action = self.spinal_cord(
            obs,
            base_forces,
            reflex_gain,
            refined_intent_exec
        )

        # PPO 统计量（只基于 raw_intent）
        log_prob = dist.log_prob(raw_intent).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()

        return (
            final_action,
            log_prob,
            value,
            entropy,
            next_hidden,
            raw_intent,
            sensory_state     # ✅ 同一次前向的感觉表征
        )

    def get_action_deterministic(self, obs, hidden=None):
        """确定性接口 (用于测试/录像)"""
        with torch.no_grad():
            sensory_state, next_hidden = self.sensory_cortex(obs, hidden)
            intent_mu, _, _ = self.motor_cortex(sensory_state) # 确定性：只取均值
            
            # 小脑前向
            cerebellum_out = self.cerebellum(intent_mu, sensory_state)
            refined = cerebellum_out[0] if isinstance(cerebellum_out, tuple) else cerebellum_out
            
            base, gain = self.brainstem(refined)
            act = self.spinal_cord(obs, base, gain, intent_mu)
        return act.cpu().numpy().flatten(), next_hidden

    def learn_from_experience(self, batch):
        """
        分脑区更新参数（模拟生物不同脑区的独立可塑性）
        """
        obs_full = batch['obs']
        old_raw_intent_full = batch['raw_intent']
        old_logp_full = batch['logp']
        returns_full = batch['return']
        advantages_full = batch['advantage']
        hidden_full = batch['hidden']

        next_obs = batch.get('next_obs', obs_full)
        mask = batch.get(
            'mask',
            torch.ones(obs_full.shape[0], 1, device=obs_full.device)
        )

        # 假设 batch_size 是数据的总长度
        dataset_size = obs_full.shape[0]
        
        # 定义超参数（建议放入初始化配置）
        ppo_epochs = 4      # 推荐 4-10 次
        batch_slice = 256    # Mini-batch 大小，防止一次梯度跑偏
        
        # 2. 开启 PPO 循环 (The Memory Loop)
        for _ in range(ppo_epochs):
            
            # 生成随机索引以打乱数据
            indices = torch.randperm(dataset_size)
            
            # Mini-batch 迭代
            for start in range(0, dataset_size, batch_slice):
                end = start + batch_slice
                idx = indices[start:end]
                
                # 提取当前 Mini-batch 数据
                obs = obs_full[idx]
                old_raw_intent = old_raw_intent_full[idx]
                old_logp = old_logp_full[idx]
                adv = advantages_full[idx]
                ret = returns_full[idx]
                # 注意：RNN hidden 在打乱后处理比较麻烦，
                # 如果是 LSTM/GRU，通常只在序列开始时传入 hidden，或者忽略 hidden 的梯度传播（只作为 context）
                # 这里暂且假设 hidden 跟随 obs 索引（简化版）
                hidden = hidden_full[:, idx, :] if hidden_full is not None else None

                # ✅ 每次迭代都重新 forward，计算“当前”策略下的 logp
                (
                    final_action,
                    new_logp,  # 这里将会随着 epoch 推进而变化！
                    values,
                    entropy,
                    _,         # next_hidden 在训练循环中通常不传递给下一轮
                    raw_intent,
                    sensory_state
                ) = self.forward(
                    obs,
                    hidden=hidden,
                    action_taken=old_raw_intent
                )

                # 维度对齐
                raw_intent = raw_intent.squeeze(1) if raw_intent.dim() == 3 else raw_intent

                # 1皮层更新 (PPO 核心)
                self._update_cortex(
                    new_logp,
                    old_logp,
                    adv,
                    values,
                    ret,
                    entropy
                )

        # === 小脑 & 脑干脊髓 独立更新阶段，这里不采用minibatch 迭代 ===
        (
            final_action,
            new_logp,  # 这里将会随着 epoch 推进而变化！
            values,
            entropy,
            _,         # next_hidden 在训练循环中通常不传递给下一轮
            raw_intent,
            sensory_state
        ) = self.forward(
            obs_full,
            hidden=hidden_full,
            action_taken=old_raw_intent_full
        )
        raw_intent = raw_intent.squeeze(1) if raw_intent.dim() == 3 else raw_intent
        # 2小脑（预测 + 校正）
        self._update_cerebellum(
            obs_full,
            next_obs,
            raw_intent,
            mask
        )

        # 3脑干 + 脊髓（结构性、生物约束）
        self._update_brainstem_spinal(
            obs_full,
            raw_intent,
            sensory_state   # ✅ 与 action 来自同一次 forward
        )

    def _update_cortex(self, new_logp, old_logp, advantages, values, returns, entropy):
        """
        皮层（策略）更新：这是【战术】层面的学习。
        Echo 正在学习"为了活下去，我该产生什么样的意图"。
        """
        optimizer = self.optimizers['cortex']
        optimizer.zero_grad()
        # PPO 策略损失
        ratio = (new_logp.squeeze() - old_logp).exp()
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()

        # 价值损失
        value_loss = 0.5 * F.mse_loss(values.squeeze(), returns)
        
        # 综合损失
        total_loss = policy_loss + value_loss - 0.01 * entropy
        #print("policy loss:", policy_loss.item(), "value loss:", value_loss.item())
        # 反向传播
        total_loss.backward() 
        
        # 梯度裁剪包含脑干
        nn.utils.clip_grad_norm_(
            list(self.sensory_cortex.parameters()) + 
            list(self.motor_cortex.parameters()),
            max_norm= 0.5
        )
        optimizer.step()
        
        # 记录
        self.losses['policy'] = policy_loss.item()
        self.losses['value'] = value_loss.item()
        self.losses['entropy'] = entropy.item()

    def _update_cerebellum(self, obs, next_obs, intent, mask=None):
        """
        小脑两阶段学习（修复了时间悖论版）：
        1) 结构统一：训练与推理使用相同的修正公式，保证实战可用。
        2) 动态课程：将“动态强度”应用在 Loss 权重上，而非前向公式中。
        """
        # ==========================================
        # 0. 准备数据
        # ==========================================
        with torch.no_grad():
            current_sensory, _ = self.sensory_cortex(obs)
            next_sensory, _ = self.sensory_cortex(next_obs)
            
            # [关键] 计算“预测难度”作为 Loss 的权重
            # 如果原始预测就错得很离谱，那么这一步的修正就显得尤为重要
            temp_pred = self.cerebellum.predict_next_state(intent, current_sensory)
            raw_error = F.mse_loss(temp_pred, next_sensory, reduction='none').mean(dim=1, keepdim=True)
            
            # 归一化权重：限制在 0.1 到 5.0 之间，避免梯度爆炸
            # 误差越大，Weight 越大，强迫 Corrector 在这些时刻必须生效
            difficulty_weight = torch.clamp(10.0 * raw_error, 0.1, 5.0)

        if mask is None:
            mask = torch.ones(intent.shape[0], 1, device=intent.device)

        # ==========================================
        # 1. 训练 Predictor (预测器)
        # ==========================================
        # A. 原始意图预测
        pred_raw = self.cerebellum.predict_next_state(intent, current_sensory)
        loss_pred_raw = (F.mse_loss(pred_raw, next_sensory, reduction='none') * mask).mean()

        # B. 修正意图预测 (协同训练)
        with torch.no_grad():
            # 这里使用固定的结构，不再依赖外部系数，保证和 Inference 一致
            # 注意：Corrector 内部最后一层建议是 Tanh，这里乘 0.1 作为物理约束
            temp_correction = self.cerebellum.compute_correction(intent, current_sensory)
            refined_intent_view = intent + 0.1 * temp_correction 
            
        pred_refined_view = self.cerebellum.predict_next_state(refined_intent_view, current_sensory)
        loss_pred_refined = (F.mse_loss(pred_refined_view, next_sensory, reduction='none') * mask).mean()

        total_pred_loss = loss_pred_raw + 0.5 * loss_pred_refined

        opt_pred = self.optimizers['cerebellum_pred']
        opt_pred.zero_grad()
        total_pred_loss.backward()
        opt_pred.step()

        # ==========================================
        # 2. 训练 Corrector (校正器)
        # ==========================================
        for p in self.cerebellum.state_predictor.parameters():
            p.requires_grad = False

        # A. 计算修正
        correction = self.cerebellum.compute_correction(intent, current_sensory)
        
        # 保持与 Inference 一致的简单结构
        # 网络会学会在需要大力修正时，detach()是为了让小脑和大脑决策分离不要互相影响，目的是知识的分离。
        refined_intent = intent.detach() + 0.1 * correction

        # B. 预测并计算误差
        pred_final = self.cerebellum.predict_next_state(refined_intent, current_sensory)
        
        # C. 动态加权 Loss
        # 这里用 difficulty_weight 乘进去。
        # 含义：对于那些 Predictor 觉得很难的样本，Corrector 修正只要有一点点改善，我们就给予巨大的奖励（梯度的反义）。
        # 反之，如果本来就很准，Corrector 乱动导致 Loss 变大，惩罚虽小但由 weight 调节。
        # 实际上我们要最小化 loss，所以 weight 越大，惩罚越重，迫使 Corrector 更好地降低这些难样本的误差。
        mse_loss = F.mse_loss(pred_final, next_sensory, reduction='none')
        loss_corr = (mse_loss * difficulty_weight * mask).mean()

        opt_corr = self.optimizers['cerebellum_corr']
        opt_corr.zero_grad()
        loss_corr.backward()
        opt_corr.step()

        for p in self.cerebellum.state_predictor.parameters():
            p.requires_grad = True

        self.losses['cerebellum_pred_raw'] = loss_pred_raw.item()
        self.losses['cerebellum_pred_refined'] = loss_pred_refined.item()
        self.losses['cerebellum_corr'] = loss_corr.item()

    def _update_brainstem_spinal(self, obs, raw_intent, sensory_state):
        """
        脑干/脊髓更新：这是【生理】层面的学习。
        Echo 正在学习"如何把抽象的意图，翻译成具体的肌肉收缩"。
        """
        optimizer = self.optimizers['brainstem_spinal']
        optimizer.zero_grad()

        # 1. 这里的 intent 必须 detach！
        # 因为在这里，我们不评价意图好坏，只评价"执行得好不好"。
        target_intent = raw_intent.detach()
        
        # 脑干展开
        brainstem_force, brainstem_gain = self.brainstem(target_intent)
        
        # 脊髓执行
        final_action = self.spinal_cord(obs, brainstem_force, brainstem_gain, target_intent)

        # ==========================
        # 核心：忠诚度 & 健康度 Loss
        # ==========================
        # 1. 听话：脑干发出的力，脊髓要大概率执行（通过 sigmoid 归一化比较）
        target_action_proxy = torch.sigmoid(brainstem_force.detach()) 
        loss_follow = F.mse_loss(final_action, target_action_proxy)

        # 2. 活性：不能为了省力就不动了（防止躺平）
        synergy_std = brainstem_force.std(dim=0).mean()
        loss_active = torch.clamp(0.1 - synergy_std, min=0) 

        # 3. 节能：动作也要尽量小（物理约束）
        loss_energy = final_action.pow(2).mean()

        # 5. 身体后果一致性（关键！） ===
        # Brainstem 必须学会：我输出的 action，大概会导致怎样的感觉变化
        pred_sensory = self.brainstem.action_to_sensory(final_action)
        target_sensory = sensory_state.detach()
        loss_embodiment = F.mse_loss(pred_sensory, target_sensory)

        # 总损失
        total_reg_loss = (
            1.0   * loss_follow +
            0.5   * loss_active +
            0e-3  * loss_energy +
            1.0  * loss_embodiment
        )

        total_reg_loss.backward()
        nn.utils.clip_grad_norm_(list(self.brainstem.parameters()) + list(self.spinal_cord.parameters()), 1.0)
        optimizer.step()

        self.losses['brainstem_spinal_follow'] = loss_follow.item()
        self.losses['brainstem_spinal_active'] = loss_active.item()
        self.losses['brainstem_spinal_energy'] = loss_energy.item()
        self.losses['brainstem_spinal_embodiment'] = loss_embodiment.item()