import pickle
from learning import classifiers


if __name__ == "__main__":
    with open('data/mix_svm_mixed_dga_grouped_family_50000_59_0.pkl', 'rb') as f:
        cl = pickle.load(f)

    test_domains = ['nvjqwnne.rwth-aachen.de', 'n5px.i.07.s.sophosxl.net', 'wqxzm.com', 'ljqgjnnzemmfrwxonmvsnjluoonin.net']
    test_groups = [0, 0, 1, 1]

    res = cl.predict(test_domains)
    print(res)

    exit(0)
