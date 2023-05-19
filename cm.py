import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st
import torch
#定义获取所有预测标签
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# def get_all_preds(model, loader):
#     all_preds = torch.tensor([]).cuda()
#     model.eval()  # set model to evaluate mode
#
#     for images, labels in loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         preds = model(images).cuda()
#         all_preds = torch.cat((all_preds, preds), dim=0)#all_preds
#
#     return all_preds



# cm = confusion_matrix(train_set.targets, train_preds.cpu().argmax(dim=1))
# print(cm)


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    if cmap is None:
        cmap = plt.get_cmap('Oranges')
    # fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 6))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(ax.imshow(cm, interpolation='nearest', cmap=cmap))
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylim(len(target_names)-0.5, -0.5)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig(title + '.png', dpi=500, bbox_inches = 'tight')
    plt.show()

    # fig,ax = plt.subplots()
    # ax.tight_layout()
    # ax.ylim(len(target_names)-0.5, -0.5)
    # ax.ylabel('True labels')
    # ax.xlabel('Predicted labels')
    # plt.show()
    #
    # ax.set_ylim(len(target_names) - 0.5, -0.5)
    # ax.set_ylabel('True labels')
    # ax.set_xlabel('Predicted labels')
    # fig.tight_layout()
    # fig.savefig(title + '.png', dpi=500, bbox_inches='tight')
    # # plt.show()

    return fig


# a tuple for all the class names
# target_names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
# figure = plot_confusion_matrix(cm, target_names)
#
# st.title("Confusion Matrix for MNIST")
# st.write("This app gives vizualization of the confusion matrix from scratch. Input: y_pred and y_hat")
# st.pyplot(figure)
#
# st.write("Plot of Epoch vs Loss")
# st.pyplot(fig1)
#
# st.write("Plot of Epoch vs Total correct prediction")
# st.pyplot(fig2)