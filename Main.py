from Dependencies import *
from Nervous_system import *
from Survival import *

def main():
    # 1. 动态获取环境维度 (修复 IndexError)
    env_name = "myoLegWalk-v0"
    
    # 先实例化一个临时的 adapter 看看这个环境到底有多少肌肉
    temp_env = MyoAdapter(env_name)
    real_n_muscles = temp_env.act_dim  # 这里会自动获取到 80
    real_obs_dim = temp_env.obs_dim
    print(f"🦵 环境检测完成: {env_name}")
    print(f"   | 肌肉执行器数量: {real_n_muscles}")
    print(f"   | 观测空间维度:   {real_obs_dim}")
    
    # 显式关闭临时环境释放内存
    temp_env.close() 

    # 2. 初始化大脑
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 初始化 BioBrain (Device: {device})...")
    
    brain = BioBrain(
        n_muscles=real_n_muscles,
        obs_dim=real_obs_dim
    ).to(device)
    
    # 3. 初始化生存引擎
    survivor = SurvivalEngine(env_name, brain, run_name="Modular_BioBrain")

    # 4. 生存-学习循环
    print("🌱 系统启动，开始生存挑战...")
    MAX_GENERATIONS = 5000
    
    for generation in range(MAX_GENERATIONS):
        try:
            # 阶段1：探索世界
            # print(f"Gen {generation}: 正在采集经验...")
            batch = survivor.collect_experience(batch_size=4096)
            
            # 阶段2：神经可塑性学习
            brain.learn_from_experience(batch)
            
            # 阶段3：评估与显示 
            if generation % 10 == 0:
                avg_reward = survivor.evaluate_performance()
                print(f"Gen {generation} | Reward: {avg_reward:.2f} | Steps: {survivor.global_step}")
                
                survivor.writer.add_scalar("Survival/Reward", avg_reward, survivor.global_step)
                # 记录损失
                for loss_name, value in brain.losses.items():
                    survivor.writer.add_scalar(f"Loss/{loss_name}", value, survivor.global_step)
                
                survivor.save_checkpoint(avg_reward)
                # 视频保存比较耗时，可以降低频率，或者只在 reward 较好时保存
            if generation % 1 == 0: 
                survivor.save_video()
                    
        except KeyboardInterrupt:
            print("🛑 用户中断，正在保存...")
            survivor.save_checkpoint(0)
            break
        except Exception as e:
            print(f"❌ 发生意外错误: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()