import matplotlib.pyplot as plt


def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    #设置显示图形大小
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    #参数项如果大于25就为25
    if num > 25: num = 25
    #画出num个数字图像
    for i in range(0,num):
        #建立subgraph子图形为5行5列
        ax = plt.subplot(5,5,1+i)
        #画出subgraph子图形
        ax.imshow(images[idx],cmap='binary')
        #设置子图像title，显示标签字段
        title = "label =" + str(labels[idx])
        #如果传入了预测结果
        if len(prediction) > 0:
            #标题
            title += ",predict=" + str(prediction[idx])
        #设置子图像的标题和刻度
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        #读取下一项
        idx += 1
    plt.show()