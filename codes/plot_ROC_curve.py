
def plot_ROC_curve(model_name, y_test, y_predict):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    img_path = './static/modelresult/' + model_name + '_ROC.png'
    if os.path.isfile(img_path):
        os.remove(img_path)
    plt.savefig(img_path)  # 此处可以回传
    plt.close()
    return img_path
    # plt.show()