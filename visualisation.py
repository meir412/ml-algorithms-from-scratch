import matplotlib.pyplot as plt

def plotLoss(loss_list):
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
