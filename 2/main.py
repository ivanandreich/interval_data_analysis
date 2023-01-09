from csv import DictReader, reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

numValues = 200


def readDataFromFile(file1='Канал 1_700nm_0.03.csv', file2='Канал 2_700nm_0.03.csv'):
    ch1U = []
    ch2U = []

    with open(file1, 'r') as f:
        csv_reader = DictReader(f, delimiter=';', )
        for row in csv_reader:
            ch1U.append(float(row['мВ']))  # actually row[0]

    with open(file2, 'r') as f:
        csv_reader = DictReader(f, delimiter=';')
        for row in csv_reader:
            ch2U.append(float(row['мВ']))  # actually row[0]

    ch1U = np.asarray(ch1U, dtype=float)
    ch2U = np.asarray(ch2U, dtype=float)

    return ch1U, ch2U


def readWeightsDataFromFile(file='w.csv'):
    w1 = []
    w2 = []

    with open(file, 'r') as f:
        csv_reader = DictReader(f, delimiter=';', )
        for row in csv_reader:
            w1.append(float(row['w1']))  # actually row[0]
            w2.append(float(row['w2']))  # actually row[1]

    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    return w1, w2

def plotData(X, Y, legends, colors, xylabels, title, show=True):
    fig, ax = plt.subplots()
    for x, y, legend, color, xylabel in zip(X, Y, legends, colors, xylabels):
        ax.plot(x, y, label=legend, color=color)
        ax.set_xlabel(xylabel[0])
        ax.set_ylabel(xylabel[1])
    ax.legend()
    plt.title(title)
    if show:
        fig.show()
    return fig, ax


def buildLinearRegression(channelU):
    # cropping bad values for regression building
    start = 19
    end = 181

    arr2 = channelU[start:end]
    arr1 = np.arange(start, end).reshape((-1, 1))

    # coefficients calculation
    model = LinearRegression().fit(arr1, arr2)
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    a = model.coef_
    b = model.intercept_

    regression = a * np.arange(0, numValues) + b
    return regression, a

def buildIntervalRegression(n, a, b):

    X_IntRegr = np.empty(shape=(len(n), 2), dtype='float')
    X_IntRegr[:, 0] = a[0] * n + b[0]
    X_IntRegr[:, 1] = a[1] * n + b[1]

    return X_IntRegr

def buildFragmentLinearRegression(channelU, points):
    # cropping bad values for regression building

    regression = np.empty(1, dtype='float')
    nodrift = np.empty(1, dtype='float')

#    print('At first, regression =', regression)

    for i in range(len(points)-1):

        start = points[i] - 1
        end = points[i + 1] - 1

        arr2 = channelU[start:end+1]
        arr1 = np.arange(start, end+1).reshape((-1, 1))

        # coefficients calculation
        model = LinearRegression().fit(arr1, arr2)
#        print(f"i start = {start + 1}")
#        print(f"i end = {end + 1}")
#        print(f"intercept: {model.intercept_}")
#        print(f"slope: {model.coef_}")

        a = model.coef_
        b = model.intercept_

        regr_part = a * np.arange(start+1, end+1) + b
        regression = np.concatenate([regression, regr_part])
        nodr_part = a * np.arange(start+1, end+1)
        nodrift = np.concatenate([nodrift, nodr_part])

        del arr1
        del arr2

    regression = np.delete(regression, 0)
    nodrift = np.delete(nodrift, 0)
    regression = np.concatenate([regression, a*(end+1) + b])
    nodrift = np.concatenate([nodrift, a*(end+1)])

    return regression, nodrift

def makeIntervals(channelU, regression, a):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(len(channelU),), dtype=float)
    tolerance.fill(5e-5)

    err = np.abs(channelU - regression)
    # print('error linear =', err)
    X_inter = np.empty(shape=(len(channelU), 2), dtype='float')
    X_inter_d = np.empty(shape=(len(channelU), 2), dtype='float')

    ind = np.arange(0, numValues)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:, 0] = channelU - err - tolerance
    X_inter[:, 1] = channelU + err + tolerance
    X_inter_d[:, 0] = X_inter[:, 0] - a * ind
    X_inter_d[:, 1] = X_inter[:, 1] - a * ind

    return X_inter_d

def makeFragmentIntervals(channelU, regression, nodrift):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(len(channelU),), dtype=float)
    tolerance.fill(5e-5)

    err = np.abs(channelU - regression)
#    print('error linear =', err)
    X_inter = np.empty(shape=(len(channelU), 2), dtype='float')
    X_inter_d = np.empty(shape=(len(channelU), 2), dtype='float')

    ind = np.arange(0, numValues)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:, 0] = channelU - err - tolerance
    X_inter[:, 1] = channelU + err + tolerance
    X_inter_d[:, 0] = X_inter[:, 0] - nodrift
    X_inter_d[:, 1] = X_inter[:, 1] - nodrift

    return X_inter_d

def makeIntervalsIntRegr(channelU, w, a, numb):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(len(w),), dtype=float)
    tolerance.fill(1e-4)

    err = w * tolerance
    constErr = 5e-5

    X_inter = np.empty(shape=(len(w), 2), dtype='float')
    X_inter_d = np.empty(shape=(len(w), 2), dtype='float')
    ind = np.arange(numb+1, numb+len(w)+1)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:, 0] = channelU - err - constErr
    X_inter[:, 1] = channelU + err + constErr

    for i in range(len(channelU)):
        X_inter_d[i][0] = X_inter[i][0] - max(a[0] * ind[i], a[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a[0] * ind[i], a[1] * ind[i])

    return X_inter_d


def makeIntervalsIntRegrFragment1(channelU, w):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(numValues,), dtype=float)
    tolerance.fill(1e-4)

    err = w * tolerance
    constErr = 5e-5

    X_inter = np.empty(shape=(numValues, 2), dtype='float')
    X_inter_d = np.empty(shape=(numValues, 2), dtype='float')
    ind = np.arange(1, numValues+1)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:, 0] = channelU - err - constErr
    X_inter[:, 1] = channelU + err + constErr

    a1 = np.empty(2, dtype=float)
    a2 = np.empty(2, dtype=float)
    a3 = np.empty(2, dtype=float)
    a4 = np.empty(2, dtype=float)
    a5 = np.empty(2, dtype=float)

    a1[0] = 6.1838e-05
    a1[1] = 1.8167e-04

    a2[0] = 4.9625e-05
    a2[1] = 5.8875e-05

    a3[0] = 4.8173e-05
    a3[1] = 5.2607e-05

    a4[0] = 5.6000e-05
    a4[1] = 5.6250e-05

    a5[0] = 1.4967e-04
    a5[1] = 2.0900e-04

    for i in range(0, 16):
        X_inter_d[i][0] = X_inter[i][0] - max(a2[0] * ind[i], a2[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a2[0] * ind[i], a2[1] * ind[i])

    for i in range(16, 38):
        X_inter_d[i][0] = X_inter[i][0] - max(a3[0] * ind[i], a3[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a3[0] * ind[i], a3[1] * ind[i])

    for i in range(38, 181):
        X_inter_d[i][0] = X_inter[i][0] - max(a4[0] * ind[i], a4[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a4[0] * ind[i], a4[1] * ind[i])

    for i in range(181, 200):
        X_inter_d[i][0] = X_inter[i][0] - max(a5[0] * ind[i], a5[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a5[0] * ind[i], a5[1] * ind[i])

    return X_inter_d


def makeIntervalsIntRegrFragment2(channelU, w):
    # multiplier to extern interval for regression capture
    tolerance = np.empty(shape=(numValues,), dtype=float)
    tolerance.fill(1e-4)

    err = w * tolerance
    constErr = 5e-5

    X_inter = np.empty(shape=(numValues, 2), dtype='float')
    X_inter_d = np.empty(shape=(numValues, 2), dtype='float')
    ind = np.arange(1, numValues+1)

    # making interval array and subtracting linear dependency to make constant
    X_inter[:, 0] = channelU - err - constErr
    X_inter[:, 1] = channelU + err + constErr

    a1 = np.empty(2, dtype=float)
    a2 = np.empty(2, dtype=float)
    a3 = np.empty(2, dtype=float)
    a4 = np.empty(2, dtype=float)
    a5 = np.empty(2, dtype=float)

    a1[0] = 6.4333e-05
    a1[1] = 9.3000e-05

    a2[0] = 3.7922e-05
    a2[1] = 3.8386e-05

    a3[0] = 5.4972e-05
    a3[1] = 6.5068e-05

    a4[0] = 3.3118e-05
    a4[1] = 3.7682e-05

    a5[0] = 8.6000e-05
    a5[1] = 1.0790e-04

    for i in range(0, 10):
        X_inter_d[i][0] = X_inter[i][0] - max(a2[0] * ind[i], a2[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a2[0] * ind[i], a2[1] * ind[i])

    for i in range(10, 59):
        X_inter_d[i][0] = X_inter[i][0] - max(a3[0] * ind[i], a3[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a3[0] * ind[i], a3[1] * ind[i])

    for i in range(59, 132):
        X_inter_d[i][0] = X_inter[i][0] - max(a4[0] * ind[i], a4[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a4[0] * ind[i], a4[1] * ind[i])

    for i in range(132, 183):
        X_inter_d[i][0] = X_inter[i][0] - max(a5[0] * ind[i], a5[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a5[0] * ind[i], a5[1] * ind[i])

    for i in range(183, 200):
        X_inter_d[i][0] = X_inter[i][0] - max(a5[0] * ind[i], a5[1] * ind[i])
        X_inter_d[i][1] = X_inter[i][1] - min(a5[0] * ind[i], a5[1] * ind[i])

    return X_inter_d


# R external estimation
def externalEstimateR(X1_inter_d, X2_inter_d):
    maxd1 = max(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1],
                X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])
    mind1 = min(X1_inter_d[0][0] / X2_inter_d[0][0], X1_inter_d[0][0] / X2_inter_d[0][1],
                X1_inter_d[0][1] / X2_inter_d[0][0], X1_inter_d[0][1] / X2_inter_d[0][1])

    for i in range(1, len(X1_inter_d[0])):
        d1 = max(X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1],
                 X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
        maxd1 = max(maxd1, d1)
        d1 = min(X1_inter_d[i][0] / X2_inter_d[i][0], X1_inter_d[i][0] / X2_inter_d[i][1],
                 X1_inter_d[i][1] / X2_inter_d[i][0], X1_inter_d[i][1] / X2_inter_d[i][1])
        mind1 = min(mind1, d1)
    print("Rext1 = ", mind1)
    print("Rext2 = ", maxd1)
    return mind1, maxd1


def calcaulateJaccard(R, X1_inter_d, X2_inter_d):
    all_intervals = np.concatenate((X1_inter_d, R * X2_inter_d), axis=0)
    intersection = all_intervals[0]
    union = all_intervals[0]
    for i in range(1, len(all_intervals)):
        intersection = [max(intersection[0], all_intervals[i][0]), min(intersection[1], all_intervals[i][1])]
        union = [min(union[0], all_intervals[i][0]), max(union[1], all_intervals[i][1])]
    jc = (intersection[1] - intersection[0]) / (union[1] - union[0])
    return jc


def internalEstimateRJaccard(Rmin, Rmax, X1_inter_d, X2_inter_d):
    R_interval = np.linspace(Rmin - 0.06, Rmax + 0.06, 1500)
    Jaccars = []

    for R in R_interval:
        Jc = calcaulateJaccard(R, X1_inter_d, X2_inter_d)
        Jaccars.append(Jc)
        print('Jc =', Jc, 'R =', R)

    print('MAX Jaccard =', max(Jaccars))
    return R_interval, max(Jaccars), Jaccars, R_interval[np.argmax(Jaccars)]


def main():
    #    plt.rcParams['text.usetex'] = True
    U1, U2 = readDataFromFile()
    w1, w2 = readWeightsDataFromFile()
    num = np.arange(1, numValues+1)

    fig, ax = plotData(
        (num, num), (U1, U2), ('First channel', 'Second channel'),
        ('red', 'green'), (('n', 'mV'), ('n', 'mV')), 'Raw data')
    x1err_r = w1 * 1e-4
    x2err_r = w2 * 1e-4
    ax.errorbar(num, U1, yerr=x1err_r, color='red', label='First channel')
    ax.errorbar(num, U2, yerr=x2err_r, color='green', label='Second channel')
    fig.show()

    # Regression for all dataset
    a1 = [4.8886e-05, 6.0523e-05]
    b1 = [4.8230e-02, 4.9305e-02]
    a2 = [4.9362e-05, 5.4194e-05]
    b2 = [5.2889e-02, 5.3464e-02]
    X1 = makeIntervalsIntRegr(U1, w1, a1, 0)
    X2 = makeIntervalsIntRegr(U2, w2, a2, 0)
    x1err = X1[:, 1] - X1[:, 0]
    x2err = X2[:, 1] - X2[:, 0]
    X_IntRegr1 = buildIntervalRegression(num, a1, b1)
    X_IntRegr2 = buildIntervalRegression(num, a2, b2)

    # Regression for dataset parts
    a1 = [6.1838e-05, 1.8761e-04]
    b1 = [4.7913e-02, 4.8751e-02]
    X_IntRegr1_p1 = buildIntervalRegression(num[0:16], a1, b1)
    a2 = [4.9625e-05, 5.8875e-05]
    b2 = [4.9238e-02, 4.9505e-02]
    X_IntRegr1_p2 = buildIntervalRegression(num[16:38], a2, b2)
    a3 = [4.8173e-05, 5.2607e-05]
    b3 = [4.9063e-02, 4.9538e-02]
    X_IntRegr1_p3 = buildIntervalRegression(num[38:181], a3, b3)
    a4 = [5.6000e-05, 5.6250e-05]
    b4 = [4.8074e-02, 4.8120e-02]
    X_IntRegr1_p4 = buildIntervalRegression(num[181:193], a4, b4)
    a5 = [1.4967e-04, 2.0900e-04]
    b5 = [1.8302e-02, 3.0040e-02]
    X_IntRegr1_p5 = buildIntervalRegression(num[193:200], a5, b5)

    X_IntRegr1_p1 = np.concatenate([X_IntRegr1_p1, X_IntRegr1_p2])
    X_IntRegr1_p1 = np.concatenate([X_IntRegr1_p1, X_IntRegr1_p3])
    X_IntRegr1_p1 = np.concatenate([X_IntRegr1_p1, X_IntRegr1_p4])
    X_IntRegr1_p1 = np.concatenate([X_IntRegr1_p1, X_IntRegr1_p5])

    a1 = [6.4333e-05, 9.3000e-05]
    b1 = [5.3075e-02, 5.3222e-02]
    X_IntRegr2_p1 = buildIntervalRegression(num[0:10], a1, b1)
    a2 = [3.7922e-05, 3.8386e-05]
    b2 = [5.3563e-02, 5.3579e-02]
    X_IntRegr2_p2 = buildIntervalRegression(num[10:59], a2, b2)
    a3 = [5.4972e-05, 6.5068e-05]
    b3 = [5.1832e-02, 5.2928e-02]
    X_IntRegr2_p3 = buildIntervalRegression(num[59:132], a3, b3)
    a4 = [3.3118e-05, 3.7682e-05]
    b4 = [5.5476e-02, 5.6228e-02]
    X_IntRegr2_p4 = buildIntervalRegression(num[132:183], a4, b4)
    a5 = [8.6000e-05, 1.0790e-04]
    b5 = [4.2296e-02, 4.6523e-02]
    X_IntRegr2_p5 = buildIntervalRegression(num[183:200], a5, b5)

    X_IntRegr2_p1 = np.concatenate([X_IntRegr2_p1, X_IntRegr2_p2])
    X_IntRegr2_p1 = np.concatenate([X_IntRegr2_p1, X_IntRegr2_p3])
    X_IntRegr2_p1 = np.concatenate([X_IntRegr2_p1, X_IntRegr2_p4])
    X_IntRegr2_p1 = np.concatenate([X_IntRegr2_p1, X_IntRegr2_p5])

    # X1_b1 = makeIntervalsIntRegr(U1[0:10], w1[0:10], [6.1838e-05, 1.8761e-04], 0)  # Data without drift, part 1
    # X2_b1 = makeIntervalsIntRegr(U2[0:10], w2[0:10], [6.4333e-05, 9.3000e-05], 0)
    # X1_b2 = makeIntervalsIntRegr(U1[16:38], w1[16:38], [1.7114e-06, 4.3667e-06], 34)  # Data without drift, part 2
    # X2_b2 = makeIntervalsIntRegr(U2[16:38], w2[34:173], [7.5385e-07, 3.6087e-06], 34)
    # X1_b3 = makeIntervalsIntRegr(U1[178:200], w1[178:200], [5.1905e-06, 1.9556e-05], 178)  # Data without drift, part 3
    # X2_b3 = makeIntervalsIntRegr(U2[178:200], w2[178:200], [2.1200e-06, 1.4375e-05], 178)

    print("X_IntRegr1_p1 size =", len(X_IntRegr1_p1))
    print("X_IntRegr2_p1 size =", len(X_IntRegr2_p1))

    #Plot curve chunks
    fig, ax = plt.subplots()
    ax.errorbar(num[0:16], U1[0:16], yerr=x1err_r[0:16], color='red', label='1 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[16:38], U1[16:38], yerr=x1err_r[16:38], color='orange', label='2 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[38:181], U1[38:181], yerr=x1err_r[38:181], color='blue', label='3 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[181:193], U1[181:193], yerr=x1err_r[181:193], color='yellow', label='4 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[193:200], U1[193:200], yerr=x1err_r[193:200], color='green', label='5 part', elinewidth=0.8, capsize=4,
                capthick=1)

    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    plt.title("Channel 1")
    fig.show()

    #Plot curve chunks
    fig, ax = plt.subplots()
    ax.errorbar(num[0:10], U2[0:10], yerr=x2err_r[0:10], color='red', label='1 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[10:59], U2[10:59], yerr=x2err_r[10:59], color='orange', label='2 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[59:132], U2[59:132], yerr=x2err_r[59:132], color='blue', label='3 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[132:183], U2[132:183], yerr=x2err_r[132:183], color='yellow', label='4 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num[183:200], U2[183:200], yerr=x2err_r[183:200], color='green', label='5 part', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    plt.title("Channel 2")
    fig.show()

    X1_int = makeIntervalsIntRegrFragment1(U1, w1)
    x1int_err = X1_int[:, 1] - X1_int[:, 0]
    X2_int = makeIntervalsIntRegrFragment2(U2, w2)
    x2int_err = X2_int[:, 1] - X2_int[:, 0]

    # # Part 1 Jaccard
    # print()
    # print("PART 1")
    # RminIn = 0.9112454340467634
    # RmaxIn = 0.9179137563317887
    # JcRminIn = 0.0063992930551573625
    # JcRmaxIn = 0.0016987232518321743
    #
    # extRmin, extRmax = externalEstimateR(X1_b1, X2_b1)
    # R_int, JaccardOpt, Jaccard, Ropt = internalEstimateRJaccard(extRmin, extRmax, X1_b1, X2_b1)
    # fig, ax = plotData((R_int,),
    #                    (Jaccard,), ('Jaccard metric',), (None,),
    #                    (('$R_{21}$', 'Jaccard'),), 'Jaccard metric', show=False)
    # ax.scatter(RminIn, JcRminIn, color='red', label=f'$R_{{min}}={RminIn:.5f}$', zorder=2)
    # ax.scatter(RmaxIn, JcRmaxIn, color='red', label=f'$R_{{max}}={RmaxIn:.5f}$', zorder=2)
    # ax.scatter(Ropt, JaccardOpt, color='blue', label=f'$R_{{opt}}={Ropt:.5f}$', zorder=2)
    # ax.legend()
    # fig.show()
    #
    # fig, ax = plt.subplots()
    # # ax.plot(ind, Ch1_U, label="Channel 1 data", color="blue")
    # ax.errorbar(num[0:28], (X1_b1[:, 0] + X1_b1[:, 1]) / 2, yerr=X1_b1[:, 1] - X1_b1[:, 0], label="Interval data X1\'", elinewidth=0.8, capsize=4,
    #             capthick=1)
    # ax.errorbar(num[0:28], Ropt * (X2_b1[:, 0] + X2_b1[:, 1]) / 2, yerr=X2_b1[:, 1] - X2_b1[:, 0], label="Interval data R*X2\'", elinewidth=0.8, capsize=4, capthick=1)
    # # ax.errorbar(ind, Ch1_U, yerr=err1, label="Interval data", marker='none', linestyle='none', color="pink")
    # ax.set_xlabel('n')
    # ax.set_ylabel('X\'')
    # plt.xticks(np.arange(0, 30, 5))
    # ax.legend()
    # # ax.grid()
    # plt.title("Fragment intervals")
    # fig.show()
    #
    # # Part 2 Jaccard
    # print()
    # print("PART 2")
    # RminIn = 0.9100967169046429
    # RmaxIn = 0.9196280915368363
    # JcRminIn = 0.0024587578477879165
    # JcRmaxIn = 0.0009345505946082808
    #
    # extRmin, extRmax = externalEstimateR(X1_b2, X2_b2)
    # R_int, JaccardOpt, Jaccard, Ropt = internalEstimateRJaccard(extRmin, extRmax, X1_b2, X2_b2)
    # fig, ax = plotData((R_int,),
    #                    (Jaccard,), ('Jaccard metric',), (None,),
    #                    (('$R_{21}$', 'Jaccard'),), 'Jaccard metric', show=False)
    # ax.scatter(RminIn, JcRminIn, color='red', label=f'$R_{{min}}={RminIn:.5f}$', zorder=2)
    # ax.scatter(RmaxIn, JcRmaxIn, color='red', label=f'$R_{{max}}={RmaxIn:.5f}$', zorder=2)
    # ax.scatter(Ropt, JaccardOpt, color='blue', label=f'$R_{{opt}}={Ropt:.5f}$', zorder=2)
    # ax.legend()
    # fig.show()
    #
    # fig, ax = plt.subplots()
    # # ax.plot(ind, Ch1_U, label="Channel 1 data", color="blue")
    # ax.errorbar(num[34:173], (X1_b2[:, 0] + X1_b2[:, 1]) / 2, yerr=X1_b2[:, 1] - X1_b2[:, 0], label="Interval data X1\'", elinewidth=0.8, capsize=4,
    #             capthick=1)
    # ax.errorbar(num[34:173], Ropt * (X2_b2[:, 0] + X2_b2[:, 1]) / 2, yerr=X2_b2[:, 1] - X2_b2[:, 0], label="Interval data R*X2\'", elinewidth=0.8, capsize=4, capthick=1)
    # # ax.errorbar(ind, Ch1_U, yerr=err1, label="Interval data", marker='none', linestyle='none', color="pink")
    # ax.set_xlabel('n')
    # ax.set_ylabel('X\'')
    # plt.xticks(np.arange(30, 180, 10))
    # ax.legend()
    # # ax.grid()
    # plt.title("Fragment intervals")
    # fig.show()
    #
    # # Part 3 Jaccard
    # print()
    # print("PART 3")
    # RminIn = 0.8732514635285763
    # RmaxIn = 0.9394906113507488
    # JcRminIn = 0.0005794850042786276
    # JcRmaxIn = 0.000685584124319693
    #
    # extRmin, extRmax = externalEstimateR(X1_b3, X2_b3)
    # R_int, JaccardOpt, Jaccard, Ropt = internalEstimateRJaccard(extRmin, extRmax, X1_b3, X2_b3)
    # fig, ax = plotData((R_int,),
    #                    (Jaccard,), ('Jaccard metric',), (None,),
    #                    (('$R_{21}$', 'Jaccard'),), 'Jaccard metric', show=False)
    # ax.scatter(RminIn, JcRminIn, color='red', label=f'$R_{{min}}={RminIn:.5f}$', zorder=2)
    # ax.scatter(RmaxIn, JcRmaxIn, color='red', label=f'$R_{{max}}={RmaxIn:.5f}$', zorder=2)
    # ax.scatter(Ropt, JaccardOpt, color='blue', label=f'$R_{{opt}}={Ropt:.5f}$', zorder=2)
    # ax.legend()
    # fig.show()
    #
    # fig, ax = plt.subplots()
    # # ax.plot(ind, Ch1_U, label="Channel 1 data", color="blue")
    # ax.errorbar(num[178:200], (X1_b3[:, 0] + X1_b3[:, 1]) / 2, yerr=X1_b3[:, 1] - X1_b3[:, 0], label="Interval data X1\'", elinewidth=0.8, capsize=4,
    #             capthick=1)
    # ax.errorbar(num[178:200], Ropt * (X2_b3[:, 0] + X2_b3[:, 1]) / 2, yerr=X2_b3[:, 1] - X2_b3[:, 0], label="Interval data R*X2\'", elinewidth=0.8, capsize=4, capthick=1)
    # # ax.errorbar(ind, Ch1_U, yerr=err1, label="Interval data", marker='none', linestyle='none', color="pink")
    # ax.set_xlabel('n')
    # ax.set_ylabel('X\'')
    # plt.xticks(np.arange(175, 200, 5))
    # ax.legend()
    # # ax.grid()
    # plt.title("Fragment intervals")
    # fig.show()


    # Plot regression parts
    fig, ax = plt.subplots()
    ax.errorbar(num, U1, yerr=x1err_r, color='red', label='First channel', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num, (X_IntRegr1[:, 0] + X_IntRegr1[:, 1])/2, yerr=X_IntRegr1[:,1]-X_IntRegr1[:,0], color='orange', label ='Regression', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    plt.title("Channel 1 Regression")
    fig.show()

    fig, ax = plt.subplots()
    ax.errorbar(num, U2, yerr=x2err_r, color='blue', label='Second channel', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num, (X_IntRegr2[:, 0] + X_IntRegr2[:, 1])/2, yerr=X_IntRegr2[:,1]-X_IntRegr2[:,0], color='magenta', label ='Regression', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    plt.title("Channel 2 Regression")
    fig.show()

    fig, ax = plt.subplots()
    ax.errorbar(num, U1, yerr=x1err_r, color='red', label='First channel', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num, (X_IntRegr1_p1[:, 0] + X_IntRegr1_p1[:, 1])/2, yerr=X_IntRegr1_p1[:,1]-X_IntRegr1_p1[:,0], color='orange', label ='Regression', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    plt.title("Channel 1 Regression")
    fig.show()

    fig, ax = plt.subplots()
    ax.errorbar(num, U2, yerr=x2err_r, color='blue', label='Second channel', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.errorbar(num, (X_IntRegr2_p1[:, 0] + X_IntRegr2_p1[:, 1])/2, yerr=X_IntRegr2_p1[:,1]-X_IntRegr2_p1[:,0], color='magenta', label ='Regression', elinewidth=0.8, capsize=4,
                capthick=1)
    ax.set_xlabel('n')
    ax.set_ylabel('mV')
    plt.xticks(np.arange(0, 201, 20))
    ax.legend()
    plt.title("Channel 2 Regression")
    fig.show()

    plt.show()

    print()


if __name__ == '__main__':
    main()
