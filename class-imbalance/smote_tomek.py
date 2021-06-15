from data_preprocessing import read_preprocessed
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.metrics import precision_recall_fscore_support
from matplotlib import pyplot
from sklearn.decomposition import PCA
from numpy import where


def smote_tomek(x_train, x_test, y_train, y_test, classifier, standalone=False):
    
    # applying smote&tomek links
    sm = SMOTETomek(random_state=42, sampling_strategy='minority')
    X_resampled, y_resampled = sm.fit_resample(x_train, y_train)
    
    # fitting ml model
    classifier.fit(X_resampled, y_resampled)
    # predicting
    y_predicted = classifier.predict(x_test)
    
    # plot for SMOTE & Tomek Links
    # dropping into 2dim
    # pca = PCA(n_components=2)
    # X_vis = pca.fit_transform(X)
    # X_res, y_res = sm.fit_resample(x_train, y_train)
    # X_res_vis = pca.transform(X_res)
    #
    # # plotting original set
    # f, (ax1, ax2) = pyplot.subplots(1, 2)
    #
    # c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Not looking for job change",
    #                  alpha=0.5)
    # c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Looking for a job change",
    #                  alpha=0.5)
    # ax1.set_title('Original set')
    #
    # # plotting SMOTE & Tomek Links
    # ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
    #             label="Not looking for job change", alpha=0.5)
    # ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
    #             label="Looking for a job change", alpha=0.5)
    # ax2.set_title('SMOTE & Tomek Links')
    #
    # # shape plotting
    # for ax in (ax1, ax2):
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     ax.spines['left'].set_position(('outward', 10))
    #     ax.spines['bottom'].set_position(('outward', 10))
    #     ax.set_xlim([-100, 100])
    #     ax.set_ylim([-100, 100])
    #
    # pyplot.figlegend((c0, c1), ('Not looking for job change', 'Looking for a job change'), loc='lower center', ncol=2, labelspacing=0.)
    # pyplot.tight_layout(pad=3)
    # pyplot.show()
    
    if standalone:
        # calculate scores
        lr_probs = classifier.predict_proba(x_test)
        lr_probs = lr_probs[:, 1]
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
        lr_f1, lr_auc = f1_score(y_test, y_predicted, average='micro'), auc(lr_recall, lr_precision)
        print('Smote & Tomek Links: f1 = %.2f%% auc = %.2f%%' % (lr_f1 * 100, lr_auc * 100))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_predicted, average='micro')
        print('Smote & Tomek Links: precision = %.2f%% recall = %.2f%%' % (prec * 100, rec * 100))
        no_skill = len(y_test[y_test == 0]) / len(y_test)
    
        # Precision - Recall Curve Plot
        pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        pyplot.plot(lr_recall, lr_precision, marker='.', label='Smote & Tomek Links')
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.legend()
        pyplot.title("Precision-Recall Curve")
        pyplot.show()

    return y_predicted
