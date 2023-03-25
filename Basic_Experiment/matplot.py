# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt

# 绘制单幅图像的路径、、、
def MatplotlibRL(exp_dir, # 图片输出的路径
                 num_episodes, # 迭代数
                 means, # 平均值
                 stds, # 方差
                 alg_name, # 算法名称
                 env_name # 实验环境
                 ):
    
    iters = [i for i in range(num_episodes)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # means = means[0]
    # stds = stds[0]
    
    r1 = list(map(lambda x: x[0]-x[1], zip(means, stds)))
    r2 = list(map(lambda x: x[0]+x[1], zip(means, stds)))
    
    # ax.plot(iters, return_list, label='CT',color='r')  # 添加标签后在图例上现实线条
    
    ax.plot(iters, means, label='CAT',color='b')  # 添加标签后在图例上现实线条
    ax.fill_between(iters, r1, r2, label='CAT', color='b', alpha=0.2)  # 添加标签后在图例上现实矩形, alpha用于改变深度
    
    font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size' : 15}
    
    plt.xlim(0, len(iters)) # 设置x轴的步数(美观)
    plt.ylim(0,)
    
    ax.set_xlabel('Number of Episode', font)
    ax.set_ylabel('Average Return', font)
    
    leg = ax.legend(loc='best', ncol=3, mode="expand", shadow=False, fancybox=True,
                    # bbox_to_anchor=(0,1.0), # bbox_to_anchor 调整图例左右位置;
                    borderaxespad = -2.6, # # borderaxespad 调整图例上下位置;
                    handlelength=2, # handlelength:调整图例线条、柱体的长短
                    prop=font)#设置图例的中字体的格式
    
    leg.get_frame().set_alpha(0.1)#用于调整图例的清晰度(0-1,透明-清晰)
    # ax.legend_.remove() # 移除图例
    plt.grid(b=True, which='major', zorder=0) # 设置网格线
    
    plt.title('{} on {}'.format(alg_name, env_name))
    fig.tight_layout()
    plt.show()
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        
    # plt.savefig(os.path.join(exp_dir, 'avgreward' + '.svg'))