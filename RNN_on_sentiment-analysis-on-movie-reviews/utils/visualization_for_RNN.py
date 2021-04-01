import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties

def result_visualization(loss_list: list,
                         correct_on_test: list,
                         correct_on_train: list,
                         test_interval: int,
                         d_model: int,
                         dropout: float,
                         BATCH_SIZE: int,
                         EPOCH: int,
                         result_figure_path: str,
                         LR: float):
    my_font = fp(fname=r"font/simsun.ttc")  # 2、设置字体路径

    # 设置风格
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_on_test, color='red', label='on Test Dataset')
    ax2.plot(correct_on_train, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
                              f'最后一轮loss:{loss_list[-1]}' '\n'
                              f'最大correct：测试集:{max(correct_on_test)}% 训练集:{max(correct_on_train)}%' '    '
                              f'最大correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}' '    '
                              f'最后一轮correct：{correct_on_test[-1]}%' '\n'
                              f'd_model={d_model} drop_out={dropout}'  '\n'
                              , FontProperties=my_font)

    # 保存结果图   测试不保存图（epoch少于draw_key）
    plt.savefig(
        f'{result_figure_path}/{max(correct_on_test)}% epoch={EPOCH} batch={BATCH_SIZE} lr={LR} d_model={d_model} dropout={dropout}.png')

    # 展示图
    plt.show()

    print('正确率列表', correct_on_test)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：测试集:{max(correct_on_test)}\t 训练集:{max(correct_on_train)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_on_test[-1]}')
