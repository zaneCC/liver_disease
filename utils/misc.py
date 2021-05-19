import matplotlib.pyplot as plt

class Misc():

    def plotAccuLine(self,accuracies,color='r',dlabel=None):
        
        plt.plot(accuracies,color,label=dlabel)
        if dlabel is not None:
            plt.legend()

        
    def show(self):
        plt.show()


# if __name__ == '__main__':
#     l1 = [0.81481481,0.84615385,0.76923077,0.84615385,0.76923077,0.76923077,
#     0.80769231,0.76923077,0.76923077,0.76923077]
#     l2 = [0.81481481,0.84615385,0.84615385,0.84615385,0.73076923,0.69230769,
#     0.80769231,0.73076923,0.80769231,0.80769231]

#     t = [4,8,6,5]

#     misc = Misc()
#     misc.plotAccuLine(l1)
#     misc.show()

    # epochs = [0,1,2,3]
    # acc = [4,8,6,5]
    # loss = [3,2,1,4]
    
    # plt.plot(acc,color='r',label='acc')        # r表示红色
    # plt.plot(loss,color=(0,0,0),label='loss')  #也可以用RGB值表示颜色
    
    #####非必须内容#########
    # plt.xlabel('epochs')    #x轴表示
    # plt.ylabel('y label')   #y轴表示
    # plt.title("chart")      #图标标题表示
    # plt.legend()            #每条折线的label显示
    #######################
    # plt.savefig('test.jpg')  #保存图片，路径名为test.jpg

    plt.show()
