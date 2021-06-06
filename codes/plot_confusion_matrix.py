def plot_confusion_matrix(model_name,cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if os.path.isfile('./static/modelresult/' + model_name + '_confusion.png'):
        os.remove('./static/modelresult/'+model_name+'_confusion.png')
    plt.savefig('./static/modelresult/'+model_name+'_confusion.png') #此处可以回传
    plt.close()
    # plt.show()