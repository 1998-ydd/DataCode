import numpy as np
import pandas as pd
import scipy.linalg as sl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
import os
plt.style.use('seaborn')
plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_columns', None) #显示所有列，把行显示设置成最大
pd.set_option('display.max_rows', None) #显示所有行，把列显示设置成最大

def read_html_SP_1(path):
    """
    读取QX8800SP第 1 批EVG测量html文件数据。
    文件夹内有两个html，一个是Mark1，另一个是Mark4
    """
    # 指定文件夹路径
    folder_path = path

    # 获取文件夹中的所有文件名
    file_names = os.listdir(folder_path)

    # 创建一个字典来存储读取的Excel文件
    html_data = {'M1':[],'M2':[]}

    # 逐个读取Excel文件
    for file_name in file_names:
        # 检查文件扩展名是否为Excel文件
        if file_name.endswith('.html'):
            # 使用Pandas读取Excel文件
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_html(file_path)
            try:
                if ('MARK1' in df[0].loc[3][1]):
                    df[4]['Identifier'] = np.array([i[6:] for i in df[4]['Identifier']], dtype=int)
                    M1 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M1'
                    html_data['M1'] = M1
                elif ('MARK4' in df[0].loc[3][1]):
                    df[4]['Identifier'] = np.array([i[6:] for i in df[4]['Identifier']], dtype=int)
                    M2 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M2'
                    html_data['M2'] = M2
            except:
                continue
    return html_data





def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False
    
def calc_theta(M1X,M1Y,M2X,M2Y):
    w = 11545.5
    h = 11739
    w2 = w - M1X + M2X
    h2 = h - M1Y + M2Y
    a1 = np.arctan2(h,w)*180/np.pi
    a2 = np.arctan2(h2,w2)*180/np.pi
    return a2-a1
def calc_center(M1X,M1Y,M2X,M2Y):
    MCX = (M2X-M1X)/2+M1X
    MCY = (M2Y-M1Y)/2+M1Y
    return MCX,MCY
def angle2offset(angle):
    return 2*np.sin(angle/2*np.pi/180)*16841.364


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plt_3sigma(data_calc,Data_ID):
    theta = np.linspace(0, 2*np.pi, 100)

    radius1 = 0.533
    a1 = radius1*np.cos(theta)
    b1 = radius1*np.sin(theta)

    radius2 = 1.067
    a2 = radius2*np.cos(theta)
    b2 = radius2*np.sin(theta)

    radius3 = 1.6
    a3 = radius3*np.cos(theta)
    b3 = radius3*np.sin(theta)
    fig, ax_nstd = plt.subplots(1,2)

    ax_nstd[0].axvline(c='grey', lw=1)
    ax_nstd[0].axhline(c='grey', lw=1)

    x1, y1 = data_calc['M1X'].dropna().values,data_calc['M1Y'].dropna().values
    ax_nstd[0].scatter(x1, y1, s=3)
    for i in data_calc.dropna().index:
        ax_nstd[0].annotate(i, xy=(data_calc.loc[i,"M1X"],data_calc.loc[i,"M1Y"]),
                    xytext=(data_calc.loc[i,"M1X"],data_calc.loc[i,"M1Y"]),
                    color="k")
    ax_nstd[0].plot(a3,b3,color='r',label=r'$3\sigma$ 1.6um')
    ax_nstd[0].plot(a2,b2,color='b',label=r'$2\sigma$ 1.067um')
    ax_nstd[0].plot(a1,b1,color='g',label=r'$1\sigma$ 0.533um')

    confidence_ellipse(x1, y1, ax_nstd[0], n_std=1, linewidth=1.5,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x1, y1, ax_nstd[0], n_std=2, linewidth=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x1, y1, ax_nstd[0], n_std=3, linewidth=2,
                       label=r'$3\sigma$', edgecolor='orange', linestyle='-.')
    ax_nstd[0].set_xlim((-3,3))
    ax_nstd[0].set_ylim((-3,3))
    ax_nstd[0].set_title(fr'{Data_ID} Die1 Mark1 $3\sigma$分布')
    ax_nstd[0].legend()


    ax_nstd[1].axvline(c='grey', lw=1)
    ax_nstd[1].axhline(c='grey', lw=1)

    x2, y2 = data_calc['M4X'].dropna().values,data_calc['M4Y'].dropna().values
    ax_nstd[1].scatter(x2, y2, s=3)

    for i in data_calc.dropna().index:
        ax_nstd[1].annotate(i, xy=(data_calc.loc[i,"M4X"],data_calc.loc[i,"M4Y"]),
                              xytext=(data_calc.loc[i,"M4X"],data_calc.loc[i,"M4Y"]),
                              color="k")
    ax_nstd[1].plot(a3,b3,color='r',label=r'$3\sigma$ 1.6um')
    ax_nstd[1].plot(a2,b2,color='b',label=r'$2\sigma$ 1.067um')
    ax_nstd[1].plot(a1,b1,color='g',label=r'$1\sigma$ 0.533um')

    confidence_ellipse(x2, y2, ax_nstd[1], n_std=1, linewidth=1.5,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x2, y2, ax_nstd[1], n_std=2, linewidth=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x2, y2, ax_nstd[1], n_std=3, linewidth=2,
                       label=r'$3\sigma$', edgecolor='orange', linestyle='-.')
    ax_nstd[1].set_xlim((-3,3))
    ax_nstd[1].set_ylim((-3,3))
    ax_nstd[1].set_title(fr'{Data_ID} Mark4 $3\sigma$分布')
    ax_nstd[1].legend()

    plt.show()

def plt_data(plt_data,name):
    fig, ax = plt.subplots(3,3)
    keys = [i for i in plt_data]
    #plot_X = plt_data.index[1]
    ax[0][0].plot(plt_data.index, plt_data[keys[0]],   linestyle = '-', # 折线类型
             linewidth = 1, color = 'steelblue', # 折线颜色
             marker = 'o',  markersize = 5, # 点的形状大小
             markeredgecolor='black', # 点的边框色
             markerfacecolor='brown') # 点的填充色
    ax[0][0].xaxis.set_major_locator(MultipleLocator(5))
    ax[0][0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[0][0].set_title(keys[0])

    ax[0][1].plot(plt_data.index, plt_data[keys[2]],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='b')
    ax[0][1].xaxis.set_major_locator(MultipleLocator(5))
    ax[0][1].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[0][1].set_title(keys[2])

    ax[0][2].plot(plt_data.index, plt_data[keys[6]],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[0][2].xaxis.set_major_locator(MultipleLocator(5))
    ax[0][2].yaxis.set_major_locator(MultipleLocator(0.0015))
    ax[0][2].set_title('偏转角度')

    ax[1][0].plot(plt_data.index, plt_data[keys[1]],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[1][0].xaxis.set_major_locator(MultipleLocator(5))
    ax[1][0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[1][0].set_title(keys[1])

    ax[1][1].plot(plt_data.index, plt_data[keys[3]],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='g')
    ax[1][1].xaxis.set_major_locator(MultipleLocator(5))
    ax[1][1].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[1][1].set_title(keys[3])

    ax[1][2].plot(plt_data.index, plt_data['MCX'],   linestyle = '-', linewidth = 1, color = 'steelblue', label='MCX',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='r')
    ax[1][2].plot(plt_data.index, plt_data['MCY'],   linestyle = '-', linewidth = 1, color = 'steelblue', label='MCY',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='y')
    ax[1][2].xaxis.set_major_locator(MultipleLocator(5))
    ax[1][2].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[1][2].set_title('Mark中心偏移XY')
    ax[1][2].legend()

    ax[2][0].plot(plt_data.index, plt_data[keys[-3]],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[2][0].axhline(1.6,c='orange',ls='-.',label='Length:1.6um')
    ax[2][0].xaxis.set_major_locator(MultipleLocator(5))
    ax[2][0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[2][0].set_title(keys[-3])
    ax[2][0].legend()

    ax[2][1].plot(plt_data.index, plt_data[keys[-2]],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[2][1].axhline(1.6,c='orange',ls='-.',label='Length:1.6um')
    ax[2][1].xaxis.set_major_locator(MultipleLocator(5))
    ax[2][1].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[2][1].set_title(keys[-2])
    ax[2][1].legend()

    ax[2][2].plot(plt_data.index, plt_data['MCL'],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[2][2].axhline(1.6,c='orange',ls='-.',label='Length:1.6um')
    ax[2][2].xaxis.set_major_locator(MultipleLocator(5))
    ax[2][2].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[2][2].set_title('Mark中心偏移MCL')
    ax[2][2].legend()
    

    plt.suptitle(f'{name}：贴片数据')
    fig.tight_layout()
    plt.show()


def wafer_data_make(data_list, gap1, new_origin):
    # 该函数用于设置wafer位置信息
    '''
    :param data_list: 如['mark1','mark2','mark3','mark4']
    :param gap1: die位置间隔设置，即每个矩形的间隔设置
    :param new_origin: 原点位置设定
    :return:
    '''
    df_empty = pd.DataFrame(columns=['芯片号', 'P_X', 'P_Y'])
    Die_count = 0
    # i的最大值即为列数
    for i in range(len(data_list)):
        # j的最大值即为该列的Die个数
        for j in range(int(data_list[i][2])):
            x_coordinate = (int(data_list[i][0]) - float(new_origin[0])) * gap1
            y_coordinate = (-int(data_list[i][1]) - j + float(new_origin[1])) * gap1
            Die_count = Die_count + 1
            df_empty.loc[Die_count - 1] = [Die_count, x_coordinate, y_coordinate]
    df_empty['芯片号'] = df_empty['芯片号'].astype(int)
    wafer_data = df_empty.set_index('芯片号')
    return wafer_data
 
    
def plt_Vmap(V_data,data0,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M4L"]
    gap = 150
    gap_rec = 400
    plt.quiver(V_data['P_X'] - gap, V_data['P_Y'] + gap,
               V_data['M1X'], V_data['M1Y'],
               V_data["M1L"], cmap='coolwarm', units='xy',width = 40)  # gap(自定义): 定义箭头的起始位置（4个Mark点的位置）
    plt.quiver(V_data["P_X"] + gap, V_data["P_Y"] - gap,
               V_data["M4X"], V_data["M4Y"],
               V_data["M4L"], cmap="coolwarm", units='xy',width = 40)
    for i in data0.index:
        color = 'k'
        plt.annotate(i, xy=(data0.loc[i,"P_X"],data0.loc[i,"P_Y"]),
                     xytext=(data0.loc[i,"P_X"]-50,data0.loc[i,"P_Y"]-50),
                     color=color)
        
        rectangle = plt.Rectangle(xy=(data0.loc[i, 'P_X'] - gap_rec/2, data0.loc[i, 'P_Y'] - gap_rec/2), width=gap_rec,
                                  height=gap_rec, fill=False, color='k')
        plt.gca().add_patch(rectangle)
        
    #     if np.isnan(L1[i]) or np.isnan(L2[i]):
    #         color = 'k'
    #     elif L1[i]<1.6 and L2[i]<1.6:
    #         color = 'b'
    #     else:
    #         color = 'r'
    #     #中心点偏移
    #     plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
    #                  xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
    #                  xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+100),
    #                  color=color)
    #     #右Mark偏移
    #     plt.annotate(r"$R$:" + str(round(L2[i],3)),
    #                  xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
    #                  xytext=(V_data.loc[i,"P_X"]+100,V_data.loc[i,"P_Y"]+100),
    #                  color=color)
    #     #左Mark偏移
    #     plt.annotate(r"$L$:" + str(round(L1[i],3)),
    #                  xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
    #                  xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-100),
    #                  color=color)
    #     plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R_14"],4)), 
    #                  xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
    #                  xytext=(V_data.loc[i,"P_X"]+100,V_data.loc[i,"P_Y"]-100),
    #                  color=color)
    # plt.xlim(-3000,2500)
    # plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移,L(左下):Mark1偏移,R(右上):Mark2偏移,(单位:um)')
    plt.show()
    

def plt_XYmap(V_data,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M4L"]
    plt.quiver(V_data["P_X"],V_data["P_Y"],V_data["MCX"],V_data["MCY"],
               V_data["MCL"], cmap="coolwarm", units='xy')
    def plt_annotate(V_data,i,color):
        #中心点偏移
        plt.annotate(r"$X$:" + str(round(V_data["MCX"][i],2)),
                     xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                     color=color)
        #右Mark偏移
        plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
                     xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                     color=color)
        #左Mark偏移
        plt.annotate(r"$Y$:" + str(round(V_data["MCY"][i],2)),                         
                     xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                     color=color)
        #角度偏移
        plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                     xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                     color=color)
    for i in V_data.index:
        plt.annotate(i, xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     color="k")
        if np.isnan(L1[i]) or np.isnan(L2[i]):
            plt_annotate(V_data,i,'k')
            
        elif L1[i]<=0.707 and L2[i]<=0.707:
            plt_annotate(V_data,i,'orange')
        
        elif L1[i]<=1.13 and L2[i]<=1.13:            
            plt_annotate(V_data,i,'g')
            
        elif L1[i]<=1.6 and L2[i]<=1.6:
            plt_annotate(V_data,i,'b')
                          
        else:
            plt_annotate(V_data,i,'r')

    # plt.xlim(-2500,2500)
    # plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移矢量,X:中心点X偏移,Y:中心点Y偏移,(单位:um)')
    plt.show()