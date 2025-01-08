# game-theory 这里的图3、4、5、6都是原论文中的 
博弈论论文代码 图4
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
rounds = 20  # 游戏轮数
epsilon_f_values = [0, 0.01, 0.03, 0.1]  # 不同的误报概率
mu0 = 0.1  # 初始信念
ba = 4  # 防御者参数
ch = 2
bc = 10
ca = 12
ct = 1
e = 1

# 计算收敛的固定点 p_th
p_th_base = (ba + ch)
p_th_denominator = (ba + ch + bc + ca + 2 * e)
p_th = p_th_base / p_th_denominator

# 修正的信念更新函数
def simulate_belief_fixed(epsilon_f, rounds, mu0, p_th, p_th_base, p_th_denominator):
    mu_t = [mu0]
    for t in range(1, rounds):
        # 收敛点的修正公式：模拟不同 epsilon_f 的影响
        fixed_point = p_th - epsilon_f * (p_th_base + ct) / p_th_denominator
        mu_next = mu_t[-1] + 0.6 * (fixed_point - mu_t[-1])  # 信念的渐近收敛过程
        mu_t.append(mu_next)
    return mu_t

# 针对不同 epsilon_f 模拟
results_fixed = {epsilon_f: simulate_belief_fixed(epsilon_f, rounds, mu0, p_th, p_th_base, p_th_denominator)
                 for epsilon_f in epsilon_f_values}

# 绘制结果
plt.figure(figsize=(10, 6))
for epsilon_f, mu_t in results_fixed.items():
    plt.plot(range(1, rounds + 1), mu_t, marker='o', label=f"$\\epsilon_f = {epsilon_f}$")

# 图形设置
plt.title("不同误报概率 $\\epsilon_f$ 下信念 $\\mu_t$ 的收敛")
plt.xlabel("轮数")
plt.ylabel("信念 $\\mu_t$")
plt.ylim(0.1, 0.21)  # 根据示例图限制y轴范围
plt.legend()
plt.grid(True)
plt.show()

图3---------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 定义参数
b_s, b_p, c_c, c_p = 10, 2, 8, 1
b_a, b_c, c_a, c_h, c_t, e = 4, 10, 12, 2, 1, 1

# 计算阈值 p_th
p_th = (b_a + c_h) / (b_a + c_h + b_c + c_a + 2 * e)
print(f"阈值 p_th = {p_th:.2f}")

# 仿真设置
num_rounds = 20  # 回合数
num_simulations = 1000  # 每种情境的仿真次数

# Fig. 3: 不同初始信念下的信念收敛性（完美观测，epsilon_f = 0）
initial_beliefs_fig3 = [0.01, 0.5, 0.99]  # mu0
epsilon_f_fig3 = 0  # 完美观测

# 预分配存储矩阵
num_initial_fig3 = len(initial_beliefs_fig3)
mean_beliefs_fig3 = np.zeros((num_initial_fig3, num_rounds + 1))

# 理论收敛值
mu_converged_fig3 = p_th

# 运行仿真
for i, mu0 in enumerate(initial_beliefs_fig3):
    all_beliefs = np.zeros((num_simulations, num_rounds + 1))
    all_beliefs[:, 0] = mu0

    for sim in range(num_simulations):
        mu = mu0
        for t in range(num_rounds):
            # 防守方根据当前信念决定是否采取防御行动
            defense_action = 1 if mu >= p_th else 0  # 采取防御 or 不采取防御

            # 攻击者类型分布
            attacker_type = np.random.rand() < p_th  # 1 表示类型1，0 表示类型2

            # 攻击者根据类型选择是否攻击
            attack = 1 if attacker_type == 1 else 0

            # 观测到的攻击行为（完美观测，epsilon_f = 0）
            observed_attack = attack

            # 贝叶斯更新
            if observed_attack:
                mu = p_th  # 完美观测直接跳到 p_th
            else:
                mu = p_th  # 仍然保持 p_th

            # 存储更新后的信念
            all_beliefs[sim, t + 1] = mu

    # 计算每轮的平均信念
    mean_beliefs_fig3[i, :] = np.mean(all_beliefs, axis=0)

# 绘制 Fig. 3 信念收敛图（不同初始信念）
plt.figure(figsize=(8, 6))
colors_fig3 = ['r', 'g', 'b']
for i, mu0 in enumerate(initial_beliefs_fig3):
    plt.plot(range(num_rounds + 1), mean_beliefs_fig3[i, :], color=colors_fig3[i],
             linewidth=2, label=f"$\mu^0 = {mu0}$")

# 绘制理论收敛值
plt.axhline(y=p_th, color='k', linestyle='--', linewidth=1.5, label=f"$p_{{th}} = {p_th:.2f}$")
plt.xlabel("回合数")
plt.ylabel("信念 $\\mu^t$")
plt.title("信念 $\\mu^t$ 随回合数的收敛性（不同初始信念）")

# 添加注释
plt.text(num_rounds * 0.7, p_th + 0.05, "理论收敛值 $p_{th}$", fontsize=10, color='k')

plt.legend(loc='best')
plt.xlim([0, num_rounds])
plt.ylim([0, 1])
plt.grid(True)
plt.show()

# 输出结果
print(f"仿真完成。")
print(f"Fig. 3: 不同初始信念下的理论收敛信念均为 p_th = {p_th:.2f}。")
图5---------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便结果可重现
np.random.seed(42)

# 游戏参数
bs, bp, cc, cp = 10, 2, 8, 1  # 攻击者参数
ba, bc, ca, ch, ct, e = 4, 10, 12, 2, 1, 1  # 防御者参数

# 信念收敛阈值和混合策略概率
p_th = (ba + ch) / (ba + ch + bc + ca + 2 * e)  # 阈值
beta_t = 1 - cp / bs  # 防御者的混合策略概率

# 游戏模拟参数
num_rounds = 50  # 游戏轮次
attacker_payoffs = np.zeros(num_rounds)
defender_payoffs = np.zeros(num_rounds)

# 模拟重复博弈
for t in range(num_rounds):
    # 随机生成攻击者类型（1表示活跃攻击者，0表示被动攻击者）
    attacker_type = np.random.rand() < p_th

    # 防御者的行动（基于混合策略概率随机化）
    defender_action = np.random.rand() < beta_t

    # 根据博弈结果计算收益
    if attacker_type == 1 and defender_action == 1:  # 活跃攻击者，防御者使用蜜罐
        attacker_payoffs[t] = -cc
        defender_payoffs[t] = bc + e  # 防御者收益
    elif attacker_type == 1 and defender_action == 0:  # 活跃攻击者，防御者未使用蜜罐
        attacker_payoffs[t] = bs + e
        defender_payoffs[t] = -ca - e  # 防御者损失
    elif attacker_type == 0 and defender_action == 1:  # 被动攻击者，防御者使用蜜罐
        attacker_payoffs[t] = 0
        defender_payoffs[t] = -ch  # 防御者成本
    elif attacker_type == 0 and defender_action == 0:  # 被动攻击者，防御者未使用蜜罐
        attacker_payoffs[t] = -bs
        defender_payoffs[t] = -ba  # 防御者轻微损失

# 绘制图 5：玩家收益随博弈轮次的变化
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_rounds + 1), attacker_payoffs, marker='*', label='Attacker', color='blue', linestyle='-')
plt.plot(range(1, num_rounds + 1), defender_payoffs, marker='o', label='Defender', color='red', linestyle='-')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Repetition', fontsize=14)
plt.ylabel('Utility of players', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
图6-----------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 设置参数
num_rounds = 1000  # 仿真回合数
epsilon_values = np.linspace(0, 0.1, 5)  # 误差范围 [0, 0.1] 分为 5 个点
attacker_payoffs_f = []  # 存储攻击者针对 εf 的平均收益
defender_payoffs_f = []  # 存储防御者针对 εf 的平均收益
attacker_payoffs_m = []  # 存储攻击者针对 εm 的平均收益
defender_payoffs_m = []  # 存储防御者针对 εm 的平均收益

# 参数
ba, bc, ca, ch = 6.5, 8, 7.5, 1.5  # 防御者参数
bs, cc = 5.5, 6  # 攻击者参数

# 固定 εm 模拟 εf 的影响
for epsilon_f in epsilon_values:
    attacker_total = 0
    defender_total = 0
    for _ in range(num_rounds):
        # 随机生成攻击者类型（1表示攻击者活跃，0表示被动/正常用户）
        attacker_type = np.random.rand() < 0.5

        # εf 影响：防御者可能误判攻击者为正常用户
        defender_action = 1 if attacker_type == 1 else 0  # 防御者理想判断
        if np.random.rand() < epsilon_f:  # 误差 εf 的影响
            defender_action = 1 - defender_action  # 反转防御者的判断

        # 收益计算
        if attacker_type == 1 and defender_action == 1:  # 活跃攻击者，防御者使用蜜罐
            attacker_total -= cc
            defender_total += bc
        elif attacker_type == 1 and defender_action == 0:  # 活跃攻击者，防御者未采取行动
            attacker_total += bs
            defender_total -= ca
        elif attacker_type == 0 and defender_action == 1:  # 正常用户被误判为攻击者
            attacker_total += 0
            defender_total -= ch
        elif attacker_type == 0 and defender_action == 0:  # 正常用户被正确判断
            attacker_total += 0
            defender_total += 0

    # 计算平均收益
    attacker_payoffs_f.append(attacker_total / num_rounds)
    defender_payoffs_f.append(defender_total / num_rounds)

# 固定 εf 模拟 εm 的影响
for epsilon_m in epsilon_values:
    attacker_total = 0
    defender_total = 0
    for _ in range(num_rounds):
        # 随机生成攻击者类型（1表示攻击者活跃，0表示被动/正常用户）
        attacker_type = np.random.rand() < 0.5

        # εm 影响：防御者可能误判正常用户为攻击者
        defender_action = 1 if attacker_type == 1 else 0  # 防御者理想判断
        if np.random.rand() < epsilon_m:  # 误差 εm 的影响
            defender_action = 1 - defender_action  # 反转防御者的判断

        # 收益计算
        if attacker_type == 1 and defender_action == 1:  # 活跃攻击者，防御者使用蜜罐
            attacker_total -= cc
            defender_total += bc
        elif attacker_type == 1 and defender_action == 0:  # 活跃攻击者，防御者未采取行动
            attacker_total += bs
            defender_total -= ca
        elif attacker_type == 0 and defender_action == 1:  # 正常用户被误判为攻击者
            attacker_total += 0
            defender_total -= ch
        elif attacker_type == 0 and defender_action == 0:  # 正常用户被正确判断
            attacker_total += 0
            defender_total += 0

    # 计算平均收益
    attacker_payoffs_m.append(attacker_total / num_rounds)
    defender_payoffs_m.append(defender_total / num_rounds)

# 绘制图表
plt.figure(figsize=(12, 8))

# 子图 (a) 和 (b)：εm = 0.05，εf 变化
plt.subplot(2, 2, 1)
plt.plot(epsilon_values, attacker_payoffs_f, marker='o', label='Attacker', color='blue')
plt.xlabel('$\epsilon_f$')
plt.ylabel('Average payoffs')
plt.title('(a) Attacker: $\epsilon_m=0.05$')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epsilon_values, defender_payoffs_f, marker='o', label='Defender', color='green')
plt.xlabel('$\epsilon_f$')
plt.ylabel('Average payoffs')
plt.title('(b) Defender: $\epsilon_m=0.05$')
plt.grid(True)
plt.legend()

# 子图 (c) 和 (d)：εf = 0.05，εm 变化
plt.subplot(2, 2, 3)
plt.plot(epsilon_values, attacker_payoffs_m, marker='o', label='Attacker', color='blue')
plt.xlabel('$\epsilon_m$')
plt.ylabel('Average payoffs')
plt.title('(c) Attacker: $\epsilon_f=0.05$')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epsilon_values, defender_payoffs_m, marker='o', label='Defender', color='green')
plt.xlabel('$\epsilon_m$')
plt.ylabel('Average payoffs')
plt.title('(d) Defender: $\epsilon_f=0.05$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
