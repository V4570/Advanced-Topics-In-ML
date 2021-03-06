from imblearn.combine import SMOTETomek
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.metrics import precision_recall_fscore_support
from matplotlib import pyplot
from sklearn.decomposition import PCA


def smote_tomek(x_train, x_test, y_train, y_test, classifier, standalone=False):
    
    # applying smote&tomek links
    sm = SMOTETomek(random_state=42, sampling_strategy='minority')
    X_resampled, y_resampled = sm.fit_resample(x_train, y_train)
    
    # fitting ml model
    classifier.fit(X_resampled, y_resampled)
    # predicting
    y_predicted = classifier.predict(x_test)
    
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


def create_graph(x_train, y_train):
    # plot for SMOTE & Tomek Links
    # dropping into 2dim
    sm = SMOTETomek(random_state=42, sampling_strategy='minority')
    X_resampled, y_resampled = sm.fit_resample(x_train, y_train)
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(x_train)
    X_res, y_res = sm.fit_resample(x_train, y_train)
    X_res_vis = pca.transform(X_res)
    #
    # plotting original set

    f, (ax1, ax2) = pyplot.subplots(1, 2)
    A = X_vis[y_train == 0, 0]
    B = X_vis[y_train == 0, 1]
    C = X_vis[y_train == 1, 0]
    D = X_vis[y_train == 1, 1]
    c0 = ax1.scatter(A[:200], B[:200], label="Not looking for job change",
                     alpha=0.5)
    c1 = ax1.scatter(C[:66], D[:66], label="Looking for a job change",
                     alpha=0.5)
    ax1.set_title('Original set')
    #
    # plotting SMOTE & Tomek Links
    A_r = X_res_vis[y_resampled == 0, 0]
    B_r = X_res_vis[y_resampled == 0, 1]
    C_r = X_res_vis[y_resampled == 1, 0]
    D_r = X_res_vis[y_resampled == 1, 1]
    ax2.scatter(A_r[:200], B_r[:200],
                label="Not looking for job change", alpha=0.5)
    ax2.scatter(C_r[:200], D_r[:200],
                label="Looking for a job change", alpha=0.5)
    ax2.set_title('SMOTE & Tomek Links')
    #
    # # shape plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-50, 50])
        ax.set_ylim([-1, 1])

    pyplot.figlegend((c0, c1), ('Not looking for job change', 'Looking for a job change'), loc='lower center', ncol=2,
                     labelspacing=0.)
    pyplot.tight_layout(pad=3)
    pyplot.show()
    pyplot.savefig('vis.png')
