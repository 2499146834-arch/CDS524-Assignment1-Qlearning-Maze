"""
双击 run_demo.bat 运行 — Q-Learning 迷宫求解器交互演示
"""
import sys
import os

# 确保工作目录是脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from trainer import Trainer
from visualizer import run_visualization

# 加载训练好的 Q 表
q_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'improved_q_table.npy')
t = Trainer(maze_size=10, seed=42)
success = t.agent.load(q_path)

if success:
    print("Q 表加载成功！")
else:
    print("未找到 Q 表，开始训练（约 15 秒）...")
    t.train(max_episodes=5000, verbose=True)
    t.agent.save(q_path)
    print("训练完成！")

print("按键: R=重置  P=暂停  Q=退出")
run_visualization(t, fps=10)
