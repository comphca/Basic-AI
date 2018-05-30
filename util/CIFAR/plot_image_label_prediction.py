import matplotlib.pyplot as plt

label_dict = {0:"airplan",1:"automobile",2:"bird",3:"cat",4:"deer",
              5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(8,14)
    if num>25: num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')

        title = str(i)+":"+label_dict[labels[i][0]]
        if len(prediction) > 0:
            title+='=>'+label_dict[prediction[i]]

        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1

    plt.show()