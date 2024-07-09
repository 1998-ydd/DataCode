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


def read_Mark1234(Data_ID):
    """
    客户4合1芯片数据读取，
    Die1有4个Mark，
    Die2右3个Mark。
    """
    folder_path = Data_ID
    # 获取html文件名
    file_names = os.listdir(Data_ID)
    html_files = []
    for file_name in file_names:
        if file_name.endswith('.html'):
            html_files.append(file_name)
    #Mark1 2 3 4，分别为Die1的Mark 1 2和Die2的Mark 1 2
    html_M1 = {'M1':[]}
    html_M2 = {'M2':[]}
    html_M3 = {'M3':[]}
    html_M4 = {'M4':[]}
    html_M = [html_M1, html_M2, html_M3, html_M4]
    # 逐个读取html文件
    for file_name in html_files:
        #组合文件路径
        file_path = os.path.join(folder_path,file_name)
            #读取文件
        df = pd.read_html(file_path)
        #通过Recipe参数定位Mark标识
        ID = ""
        try:
            if ('MARK1' in df[0].loc[3][1]):
                html_file = html_M[0]
                ID = "M1"
            elif ('MARK2' in df[0].loc[3][1]):
                html_file = html_M[1]
                ID = "M2"
            elif ('MARK3' in df[0].loc[3][1]):
                html_file = html_M[2]
                ID = "M3"
            elif ('MARK4' in df[0].loc[3][1]):
                html_file = html_M[3]
                ID = "M4"
        except:
            print("Error: Recipe参数错误！")

        #定位想要获取的数据
        try:
            df[4]['Identifier'] = np.array([i[3:] for i in df[4]['Identifier']], dtype=int)
        except:
            df[4]['Identifier'] = np.array([i for i in df[4]['Identifier']], dtype=int)
        M = df[4].iloc[:,[0,5,6]].set_index('Identifier')        
        html_file[ID] = M
    return html_M

def read_Mark5678(Data_ID):
    folder_path = Data_ID
    # 获取html文件名
    file_names = os.listdir(Data_ID)
    html_files = []
    for file_name in file_names:
        if file_name.endswith('.html'):
            html_files.append(file_name)
    #Mark5 6 7 8，分别为Die1的Mark 1 2和Die2的Mark 1 2
    html_M5 = {'M1':[]}
    html_M6 = {'M1':[]}
    html_M7 = {'M1':[]}
    html_M8 = {'M1':[]}
    html_M = [html_M5, html_M6, html_M7, html_M8]
    # 逐个读取html文件
    for file_name,html_file in zip(html_files,html_M):
		#组合文件路径
        file_path = os.path.join(folder_path,file_name)
            #读取文件
        df = pd.read_html(file_path)
            #定位想要获取的数据
        df[4]['Identifier'] = np.array([i[3:] for i in df[4]['Identifier']], dtype=int)
        M1 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
        #ID = df[0].loc[0][1]
        html_file['M1'] = M1
    return html_M

def read_data_allD1(path):
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
                if ('Mark5' in df[0].loc[2][1]):
                    df[4]['Identifier'] = np.array([i[3:] for i in df[4]['Identifier']], dtype=int)
                    M1 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M1'
                    html_data['M1'] = M1
                elif ('Mark6' in df[0].loc[2][1]):
                    df[4]['Identifier'] = np.array([i[3:] for i in df[4]['Identifier']], dtype=int)
                    M2 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M2'
                    html_data['M2'] = M2
            except:
                continue
    return html_data

def read_data_allD2(path):
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
                if ('Mark7' in df[0].loc[2][1]):
                    df[4]['Identifier'] = np.array([i[3:] for i in df[4]['Identifier']], dtype=int)
                    M1 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M1'
                    html_data['M1'] = M1
                elif ('Mark8' in df[0].loc[2][1]):
                    df[4]['Identifier'] = np.array([i[3:] for i in df[4]['Identifier']], dtype=int)
                    M2 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M2'
                    html_data['M2'] = M2
            except:
                continue
    return html_data
def read_data_allD2_2(path):
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
                if ('Mark7' in df[0].loc[2][1]):
                    df[4]['Identifier'] = np.array([i[3:] for i in df[4]['Identifier']], dtype=int)
                    M1 = df[4].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M1'
                    html_data['M1'] = M1
                elif ('Mark8' in df[0].loc[2][1]):
                    df[2]['Identifier'] = np.array([i[3:] for i in df[2]['Identifier']], dtype=int)
                    M2 = df[2].iloc[::-1,[0,5,6]].set_index('Identifier')
                    ID = df[0].loc[0][1]
                    file_name = ID + 'M2'
                    html_data['M2'] = M2
            except:
                continue
    return html_data
def read_data(path):
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
                if ('Mark7' in df[0].loc[11][1]):
                    M1 = df[2].iloc[:,:3].set_index(('Box in Box','#'))
                    ID = df[0].loc[4][1]
                    file_name = ID + 'M1'
                    html_data['M1'] = M1
                elif ('Mark8' in df[0].loc[11][1]):
                    M2 = df[2].iloc[:,:3].set_index(('Box in Box','#'))
                    ID = df[0].loc[4][1]
                    html_data['M2'] = M2
            except:
                continue
    return html_data

def read_data1(path):
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
                if ('MARK5' in df[0].loc[11][1]):
                    M1 = df[2].iloc[:,:3].set_index(('Box in Box','#'))
                    ID = df[0].loc[4][1]
                    html_data['M1'] = M1
                elif ('MARK6' in df[0].loc[11][1]):
                    M2 = df[2].iloc[:,:3].set_index(('Box in Box','#'))
                    ID = df[0].loc[4][1]
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

    x2, y2 = data_calc['M2X'].dropna().values,data_calc['M2Y'].dropna().values
    ax_nstd[1].scatter(x2, y2, s=3)

    for i in data_calc.dropna().index:
        ax_nstd[1].annotate(i, xy=(data_calc.loc[i,"M2X"],data_calc.loc[i,"M2Y"]),
                              xytext=(data_calc.loc[i,"M2X"],data_calc.loc[i,"M2Y"]),
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
    ax_nstd[1].set_title(fr'{Data_ID} Die1 Mark2 $3\sigma$分布')
    ax_nstd[1].legend()

    plt.show()

def plt_data(plt_data,name):
    fig, ax = plt.subplots(3,3)
    #plot_X = plt_data.index[1]
    ax[0][0].plot(plt_data.index, plt_data['M1X'],   linestyle = '-', # 折线类型
             linewidth = 1, color = 'steelblue', # 折线颜色
             marker = 'o',  markersize = 5, # 点的形状大小
             markeredgecolor='black', # 点的边框色
             markerfacecolor='brown') # 点的填充色
    ax[0][0].xaxis.set_major_locator(MultipleLocator(5))
    ax[0][0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[0][0].set_title('Mark1偏移M1X')

    ax[0][1].plot(plt_data.index, plt_data['M2X'],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='b')
    ax[0][1].xaxis.set_major_locator(MultipleLocator(5))
    ax[0][1].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[0][1].set_title('Mark2偏移M2X')

    ax[0][2].plot(plt_data.index, plt_data['M_R'],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[0][2].xaxis.set_major_locator(MultipleLocator(5))
    ax[0][2].yaxis.set_major_locator(MultipleLocator(0.0015))
    ax[0][2].set_title('偏转角度')

    ax[1][0].plot(plt_data.index, plt_data['M1Y'],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[1][0].xaxis.set_major_locator(MultipleLocator(5))
    ax[1][0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[1][0].set_title('Mark1偏移M1Y')

    ax[1][1].plot(plt_data.index, plt_data['M2Y'],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='g')
    ax[1][1].xaxis.set_major_locator(MultipleLocator(5))
    ax[1][1].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[1][1].set_title('Mark2偏移M2Y')

    ax[1][2].plot(plt_data.index, plt_data['MCX'],   linestyle = '-', linewidth = 1, color = 'steelblue', label='MCX',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='r')
    ax[1][2].plot(plt_data.index, plt_data['MCY'],   linestyle = '-', linewidth = 1, color = 'steelblue', label='MCY',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='y')
    ax[1][2].xaxis.set_major_locator(MultipleLocator(5))
    ax[1][2].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[1][2].set_title('Mark中心偏移XY')
    ax[1][2].legend()

    ax[2][0].plot(plt_data.index, plt_data['M1L'],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[2][0].axhline(1.6,c='orange',ls='-.',label='Length:1.6um')
    ax[2][0].xaxis.set_major_locator(MultipleLocator(5))
    ax[2][0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[2][0].set_title('Mark1偏移M1L')
    ax[2][0].legend()

    ax[2][1].plot(plt_data.index, plt_data['M2L'],   linestyle = '-', linewidth = 1, color = 'steelblue',
             marker = 'o',  markersize = 5, markeredgecolor='black',  markerfacecolor='m')
    ax[2][1].axhline(1.6,c='orange',ls='-.',label='Length:1.6um')
    ax[2][1].xaxis.set_major_locator(MultipleLocator(5))
    ax[2][1].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[2][1].set_title('Mark2偏移M2L')
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

def plt_Vmap68(V_data,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M2L"]
    plt.quiver(V_data["P_X"],V_data["P_Y"],V_data["MCX"],V_data["MCY"],
               V_data["MCL"], cmap="coolwarm", units='xy')
    for i in V_data.index:
        plt.annotate(i, xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     color="k")
        if np.isnan(L1[i]) or np.isnan(L2[i]):
            #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="k")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="k")
        elif L1[i]<1.6 and L2[i]<1.6:
            #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="b")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="b")
        else:
             #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="r")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="r")
    plt.xlim(0,2500)
    plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移,L(左下):Mark1偏移,R(右上):Mark2偏移,(单位:um)')
    plt.show()
    
def plt_Vmap69(V_data,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M2L"]
    plt.quiver(V_data["P_X"],V_data["P_Y"],V_data["MCX"],V_data["MCY"],
               V_data["MCL"], cmap="coolwarm", units='xy')
    for i in V_data.index:
        plt.annotate(i, xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     color="k")
        if np.isnan(L1[i]) or np.isnan(L2[i]):
            #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="k")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="k")
        elif L1[i]<1.6 and L2[i]<1.6:
            #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="b")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="b")
        else:
             #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="r")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="r")
    plt.xlim(-3000,1000)
    plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移,L(左下):Mark1偏移,R(右上):Mark2偏移,(单位:um)')
    plt.show()
    
def plt_Vmap157(V_data,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M2L"]
    plt.quiver(V_data["P_X"],V_data["P_Y"],V_data["MCX"],V_data["MCY"],
               V_data["MCL"], cmap="coolwarm", units='xy')
    for i in V_data.index:
        plt.annotate(i, xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     color="k")
        if np.isnan(L1[i]) or np.isnan(L2[i]):
            #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="k")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="k")
        elif L1[i]<1.6 and L2[i]<1.6:
            #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="b")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="b")
        else:
             #中心点偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #右Mark偏移
            plt.annotate(r"$R$:" + str(round(L2[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #左Mark偏移
            plt.annotate(r"$L$:" + str(round(L1[i],3)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="r")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="r")
    plt.xlim(-3000,2500)
    plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移,L(左下):Mark1偏移,R(右上):Mark2偏移,(单位:um)')
    plt.show()
    
def plt_XYmap68(V_data,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M2L"]
    plt.quiver(V_data["P_X"],V_data["P_Y"],V_data["MCX"],V_data["MCY"],
               V_data["MCL"], cmap="coolwarm", units='xy')
    for i in V_data.index:
        plt.annotate(i, xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     color="k")
        if np.isnan(L1[i]) or np.isnan(L2[i]):
             #中心点偏移
            plt.annotate(r"$X$:" + str(round(V_data["MCX"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #右Mark偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #左Mark偏移
            plt.annotate(r"$Y$:" + str(round(V_data["MCY"][i],2)),                         
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="k")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="k")
        elif L1[i]<1.6 and L2[i]<1.6:
            #中心点偏移
            plt.annotate(r"$X$:" + str(round(V_data["MCX"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #右Mark偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #左Mark偏移
            plt.annotate(r"$Y$:" + str(round(V_data["MCY"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="b")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="b")

        else:
                         #中心点偏移
            plt.annotate(r"$X$:" + str(round(V_data["MCX"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #右Mark偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #左Mark偏移
            plt.annotate(r"$Y$:" + str(round(V_data["MCY"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="r")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="r")
    plt.xlim(0,2500)
    plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移矢量,X:中心点X偏移,Y:中心点Y偏移,(单位:um)')
    plt.show()

def plt_XYmap69(V_data,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M2L"]
    plt.quiver(V_data["P_X"],V_data["P_Y"],V_data["MCX"],V_data["MCY"],
               V_data["MCL"], cmap="coolwarm", units='xy')
    for i in V_data.index:
        plt.annotate(i, xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     xytext=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                     color="k")
        if np.isnan(L1[i]) or np.isnan(L2[i]):
             #中心点偏移
            plt.annotate(r"$X$:" + str(round(V_data["MCX"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #右Mark偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="k")
            #左Mark偏移
            plt.annotate(r"$Y$:" + str(round(V_data["MCY"][i],2)),                         
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="k")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="k")
        elif L1[i]<1.6 and L2[i]<1.6:
            #中心点偏移
            plt.annotate(r"$X$:" + str(round(V_data["MCX"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #右Mark偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="b")
            #左Mark偏移
            plt.annotate(r"$Y$:" + str(round(V_data["MCY"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="b")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="b")

        else:
                         #中心点偏移
            plt.annotate(r"$X$:" + str(round(V_data["MCX"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #右Mark偏移
            plt.annotate(r"$C$:" + str(round(V_data["MCL"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]+50),
                         color="r")
            #左Mark偏移
            plt.annotate(r"$Y$:" + str(round(V_data["MCY"][i],2)),
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]-200,V_data.loc[i,"P_Y"]-50),
                         color="r")
            plt.annotate(r"$\theta$:" + str(round(V_data.loc[i,"M_R"],4)), 
                         xy=(V_data.loc[i,"P_X"],V_data.loc[i,"P_Y"]),
                         xytext=(V_data.loc[i,"P_X"]+50,V_data.loc[i,"P_Y"]-50),
                         color="r")
    plt.xlim(-2500,500)
    plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移矢量,X:中心点X偏移,Y:中心点Y偏移,(单位:um)')
    plt.show()
    
def plt_XYmap157(V_data,name):
    plt.figure()
    L1 = V_data["M1L"]
    L2 = V_data["M2L"]
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

    plt.xlim(-2500,2500)
    plt.ylim(-3000,3000)
    plt.colorbar()
    plt.title(f'{name}：C:中心点偏移矢量,X:中心点X偏移,Y:中心点Y偏移,(单位:um)')
    plt.show()