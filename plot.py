import matplotlib.pyplot as plt
import numpy as np

# sum_of_planes_plot.txtのデータを読み込む
theta = []

CS_SrO = []
CS2_SrO = []

CS_TiO = []
CS2_TiO = []

#with open('SrO.txt', 'r') as file:
#    # 1?~L?~[??~B~R?~B??~B??~C~C?~C~W
#    next(file)
#    for line in file:
#        data = line.split()
#        theta.append(float(data[1]))
#        CS_SrO.append(float(data[3]))
##
#for i in range(len(theta)):
#    CS2_SrO.append(CS_SrO[i]/(CS_SrO[8])+0.2)

with open('TiO2.txt', 'r') as file:
    # 1?~L?~[??~B~R?~B??~B??~C~C?~C~W
    next(file)
    for line in file:
        data = line.split()
        theta.append(float(data[1]))
        CS_TiO.append(float(data[3]))
#
for i in range(len(theta)):
    CS2_TiO.append(CS_TiO[i]/(CS_TiO[8])+0.2)

signal_ex = []
Ex = []
#
with open('Ex.txt', 'r') as file:
#    # 1行目をスキップ
    next(file)
    for line in file:
        data = line.split()
        signal_ex.append(float(data[0]))  # 1列目のデータ (cross_section)
#
for i in range(len(signal_ex)):
    Ex.append(signal_ex[i])
#
plt.figure(figsize=(8, 8))

# 実験データ
plt.plot(theta, Ex,
         color='#000000',       # 黒
         linewidth=1.5,
         linestyle='-',         # 実線
         marker='o',            # 塗りつぶし丸
         markersize=6,
         markerfacecolor='black',
         markeredgecolor='black',
         label='Experiment')

# シミュレーションデータ
#plt.plot(theta, CS2_SrO,
#         color='#D62728',       # 赤（白黒印刷時は濃灰にしてもOK）
#         linewidth=1.5,
#         linestyle='--',        # 破線
#         marker='o',            # 中抜き丸
#         markersize=6,
#         markerfacecolor='white',
#         markeredgecolor='#D62728',
#         label='Series Expansion (SrO Termination)')

plt.plot(theta, CS2_TiO,
         color='blue',       # 赤（白黒印刷時は濃灰にしてもOK）
         linewidth=1.5,
         linestyle='--',        # 破線
         marker='o',            # 中抜き丸
         markersize=6,
         markerfacecolor='white',
         markeredgecolor='blue',
         label='Series Expansion (TiO2 Termination)')

# 軸範囲
#plt.ylim(0.5, 1.5)

# y軸目盛り非表示
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
ticks = np.arange(-10,80,10)
plt.xticks(ticks)

# 軸ラベル
plt.xlabel('Polar (°)', fontsize=23)
plt.ylabel('Intensity (arb. units)', fontsize=23)

# 凡例
#plt.legend(fontsize=15)

# x軸のラベルサイズ
plt.tick_params(axis='x', which='major', labelsize=22)

# グラフ表示
plt.show()
