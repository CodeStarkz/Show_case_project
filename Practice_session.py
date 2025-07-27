import numpy as np
import scipy.io
from scipy.stats import kurtosis, skew
from scipy.stats import zscore
from scipy.signal import find_peaks

P = 51
w1 = {}
Se_no = np.zeros((P, 8))
Thresolden_no = np.zeros((P, 8))
normen_no = np.zeros((P, 8))
sureen_no = np.zeros((P, 8))
ivag_no = np.zeros((P, 8))
mav_no = np.zeros((P, 8))
ssi_no = np.zeros((P, 8))
approen_no = np.zeros((P, 8))
Le_no = np.zeros((P, 8))
k_no = np.zeros((P, 8))
sk_no = np.zeros((P, 8))
m_no = np.zeros((P, 8))
r_no = np.zeros((P, 8))
s_no = np.zeros((P, 8))
m2_no = np.zeros((P, 8))

for i in range(1, P + 1):
    fname = f'novag{i}.mat'
    X = scipy.io.loadmat(fname)
    s1 = X['data'].flatten()

    NN = len(s1)
    p = 275
    q = 279
    r = 1
    s = 10
    bet = 0.95 * r / s
    J = 7
    F = CreateFilters(NN, p, q, r, s, bet, J)
    w1[i] = RAnDwt(s1, p, q, r, s, J, F)

    for p in range(8):
        ll = np.abs(w1[i][p])

        Se_no[i - 1, p] = wentropy(ll, 'shannon')
        Thresolden_no[i - 1, p] = wentropy(ll, 'threshold', 0.2)
        normen_no[i - 1, p] = wentropy(ll, 'norm', 1.1)
        sureen_no[i - 1, p] = wentropy(ll, 'sure', 3)

        ivag_no[i - 1, p] = np.sum(np.abs(ll))
        mav_no[i - 1, p] = np.sum(np.abs(ll)) / len(ll)
        ssi_no[i - 1, p] = np.sum(np.abs(ll) ** 2)
        approen_no[i - 1, p] = approximateEntropy(ll)
        Le_no[i - 1, p] = wentropy(ll, 'log energy')
        k_no[i - 1, p] = kurtosis(ll)
        sk_no[i - 1, p] = skew(ll)
        m_no[i - 1, p] = np.mean(ll)
        r_no[i - 1, p] = np.sqrt(np.mean(ll ** 2))
        s_no[i - 1, p] = np.std(ll)
        m2_no[i - 1, p] = np.median(ll)

w2 = {}
for i in range(1, 39):
    fname = f'abvag{i}.mat'
    X = scipy.io.loadmat(fname)
    s2 = X['data'].flatten()
    s3 = s2.data.flatten()

    NM = len(s3)
    p = 275
    q = 279
    r = 1
    s = 10
    bet = 0.95 * r / s
    J = 7
    F = CreateFilters(NM, p, q, r, s, bet, J)
    w2[i] = RAnDwt(s3, p, q, r, s, J, F)

Se_ab = np.zeros((38, 8))
Thresolden_ab = np.zeros((38, 8))
normen_ab = np.zeros((38, 8))
sureen_ab = np.zeros((38, 8))
ivag_ab = np.zeros((38, 8))
mav_ab = np.zeros((38, 8))
ssi_ab = np.zeros((38, 8))
approen_ab = np.zeros((38, 8))
Le_ab = np.zeros((38, 8))
k_ab = np.zeros((38, 8))
sk_ab = np.zeros((38, 8))
m_ab = np.zeros((38, 8))
r_ab = np.zeros((38, 8))
s_ab = np.zeros((38, 8))
m2_ab = np.zeros((38, 8))

for i in range(1, 39):
    for m in range(8):
        llm = np.abs(w2[i][m])

        Se_ab[i - 1, m] = wentropy(llm, 'shannon')
        Thresolden_ab[i - 1, m] = wentropy(llm, 'threshold', 0.2)
        normen_ab[i - 1, m] = wentropy(llm, 'norm', 1.1)
        sureen_ab[i - 1, m] = wentropy(llm, 'sure', 3)

        ivag_ab[i - 1, m] = np.sum(np.abs(llm))
        mav_ab[i - 1, m] = np.sum(np.abs(llm)) / len(llm)
        ssi_ab[i - 1, m] = np.sum(np.abs(llm) ** 2)
        approen_ab[i - 1, m] = approximateEntropy(llm)
        Le_ab[i - 1, m] = wentropy(llm, 'log energy')
        k_ab[i - 1, m] = kurtosis(llm)
        sk_ab[i - 1, m] = skew(llm)
        m_ab[i - 1, m] = np.mean(llm)
        r_ab[i - 1, m] = np.sqrt(np.mean(llm ** 2))
        s_ab[i - 1, m] = np.std(llm)
        m2_ab[i - 1, m] = np.median(llm)

k = m2_no[0:38, 7]
KW1_Test = kruskalwallis(np.concatenate((k, m2_ab[:, 7])))

for i in range(1, 8):
    sam1 = np.concatenate((m_no[:, i], m_ab[:, i]))
    sam2 = np.concatenate((r_no[:, i], r_ab[:, i]))
    sam3 = np.concatenate((Le_no[:, i], Le_ab[:, i]))
    sam4 = np.concatenate((s_no[:, i], s_ab[:, i]))
    sam5 = np.concatenate((mav_no[:, i], mav_ab[:, i]))
    sam6 = np.concatenate((Thresolden_no[:, i], Thresolden_ab[:, i]))
    sam7 = np.concatenate((sureen_no[:, i], sureen_ab[:, i]))
    sam8 = np.concatenate((normen_no[:, i], normen_ab[:, i]))
    sam9 = np.concatenate((Se_no[:, i], Se_ab[:, i]))
    sam10 = np.concatenate((m_no[:, i], m_ab[:, i]))
    sam11 = np.concatenate((ssi_no[:, i], ssi_ab[:, i]))
    sam12 = np.concatenate((m2_no[:, i], m2_ab[:, i]))

blue = np.column_stack((
                       zscore(sam1), zscore(sam2), zscore(sam3), zscore(sam4), zscore(sam5), zscore(sam6), zscore(sam7),
                       zscore(sam8), zscore(sam9), zscore(sam10), zscore(sam11), zscore(sam12)))
kew = np.concatenate((np.zeros(51), np.ones(38)))
black = np.column_stack((blue, kew))

