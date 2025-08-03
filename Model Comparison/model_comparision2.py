import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullFitter, LogLogisticFitter

# 读取数据
file_path = "/Users/fujunhan/Desktop/RA_traffic simulation/my_research_related_materials/plan1.11/parking_data.xlsx"
df = pd.read_excel(file_path)

# 划分训练集和测试集
train_data = df.iloc[:800]
test_data = df.iloc[800:]

# 设置时间截断阈值
time_threshold = 75000

# 筛选 State = 0 数据 (空闲状态)
state_0 = train_data[(train_data['State'] == 0) & (train_data['Lasting Time'] <= time_threshold)]
durations_state_0 = state_0['Lasting Time']
events_state_0 = [1] * len(state_0)

# 筛选 State = 1 数据 (占用状态)
state_1 = train_data[(train_data['State'] == 1) & (train_data['Lasting Time'] <= time_threshold)]
durations_state_1 = state_1['Lasting Time']
events_state_1 = [1] * len(state_1)

# 初始化拟合器
kmf = KaplanMeierFitter()
weibull = WeibullFitter()
loglogistic = LogLogisticFitter()

# 创建并列子图
plt.figure(figsize=(12, 6))

# ------------------------- 左图：空闲状态 -------------------------
plt.subplot(1, 2, 1)

# 拟合空闲状态数据
kmf.fit(durations_state_0, event_observed=events_state_0, label="Kaplan-Meier")
weibull.fit(durations_state_0, event_observed=events_state_0)
loglogistic.fit(durations_state_0, event_observed=events_state_0)

# 绘制空闲状态曲线
kmf.plot_survival_function(ci_show=False, color='blue')
plt.plot(weibull.survival_function_.index, weibull.survival_function_.values, 
         color='green', label="Weibull")
plt.plot(loglogistic.survival_function_.index, loglogistic.survival_function_.values, 
         color='red', label="Log-Logistic")

plt.title("Idle State (State = 0)", fontsize=14, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=12, fontweight="bold")
plt.ylabel("Survival Probability", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)

# ------------------------- 右图：占用状态 -------------------------
plt.subplot(1, 2, 2)

# 拟合占用状态数据
kmf.fit(durations_state_1, event_observed=events_state_1, label="Kaplan-Meier")
weibull.fit(durations_state_1, event_observed=events_state_1)
loglogistic.fit(durations_state_1, event_observed=events_state_1)

# 绘制占用状态曲线
kmf.plot_survival_function(ci_show=False, color='blue')
plt.plot(weibull.survival_function_.index, weibull.survival_function_.values, 
         color='green', label="Weibull")
plt.plot(loglogistic.survival_function_.index, loglogistic.survival_function_.values, 
         color='red', label="Log-Logistic")

plt.title("Occupied State (State = 1)", fontsize=14, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=12, fontweight="bold")
plt.ylabel("Survival Probability", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)

# 调整整体布局
plt.suptitle("Survival Function Comparison: Idle vs Occupied States", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()