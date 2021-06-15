from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from pool_based import pool_based, plot_accuracy, build_conf_matrix


def main():

    data = 'processed.csv'
    #clf = KNeighborsClassifier(n_neighbors=3)
    clf = AdaBoostClassifier()
    queries = 40
    for i in [uncertainty_sampling, margin_sampling, entropy_sampling]:
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------')
        query_strategy = i

        list, array_1, array_2 = pool_based(data, clf, queries, query_strategy)
        plot_accuracy(list)
        build_conf_matrix(array_1, array_2)



if __name__ == '__main__':
    main()