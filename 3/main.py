import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import ast


def get_radius(interval):
    return (max(interval) - min(interval)) / 2


def get_middle(interval):
    return (interval[0] + interval[1]) / 2


def draw_data_status_lawn(x_lims=(0, 2), title='Influences'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('lemonchiffon')
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(-(x_lims[1] + 1), x_lims[1] + 1)
    # draw green triangle zone
    x1, y1 = [0, 1], [-1, 0]
    x2, y2 = [0, 1], [1, 0]
    ax.plot(x1, y1, 'k', x2, y2, 'k')
    ax.fill_between(x1, y1, y2, facecolor='lawngreen')

    # draw others zones
    x1, y1 = [0, x_lims[1]], [-1, -(x_lims[1] + 1)]
    x2, y2 = [0, x_lims[1]], [1, x_lims[1] + 1]

    ax.plot(x1, y1, 'k', x2, y2, 'k')
    x = np.arange(0.0, x_lims[1], 0.01)
    y1 = x + 1
    y2 = [x_lims[1] + 1] * len(x)
    ax.fill_between(x, y1, y2, facecolor='salmon')
    y2 = [-(x_lims[1] + 1)] * len(x)
    ax.fill_between(x, -y1, y2, facecolor='salmon')

    x1, y1 = [1, 1], [-(x_lims[1] + 1), x_lims[1] + 1]
    ax.plot(x1, y1, 'k--')
    ax.set_xlabel('l(x, y)')
    ax.set_ylabel('r(x, y)')
    ax.set_title(title)
    return fig, ax


def add_point(point, ax):
    ax.plot(point[0], point[1], 'bo', markersize=5)


def get_intersections(interval_list):
    res = interval_list[0]
    for i in range(1, len(interval_list), 1):
        res = [max(min(res), min(interval_list[i])), min(max(res), max(interval_list[i]))]
    return res


def get_intersections_wrong_interval(interval_list):
    res = interval_list[0]
    for i in range(1, len(interval_list), 1):
        res = [max(res[0], interval_list[i][0]), min(res[1], interval_list[i][1])]
    return res


def get_influences(interval_list, intersection_=None):
    if intersection_ is not None:
        intersection = intersection_
    else:
        intersection = get_intersections(interval_list)
    inter_rad = get_radius(intersection)
    inter_mid = get_middle(intersection)
    influences = []
    for interval in interval_list:
        l = inter_rad / get_radius(interval)
        r = (get_middle(interval) - inter_mid) / get_radius(interval)
        influences.append([l, r])
    return influences, intersection


def get_residuals(interval_d, edge_points, drift_params_3):
    new_list = []
    for list_num, list_ in enumerate(interval_d):
        new_list__ = []
        for num, drift_param in enumerate(drift_params_3[list_num]):
            if num == 0:
                new_list_ = list_[:edge_points[list_num][0]]
                start = 0
            elif num == 1:
                new_list_ = list_[edge_points[list_num][0]:edge_points[list_num][1]]
                start = edge_points[list_num][0]
            else:
                new_list_ = list_[edge_points[list_num][1]:]
                start = edge_points[list_num][1]
            for num_, interval in enumerate(new_list_, start=start):
                new_list__.append([interval[0] - (num_ + 1) * drift_param[0][1] - drift_param[1][1],
                                   interval[1] - (num_ + 1) * drift_param[0][0] - drift_param[1][0]])
        new_list.append(new_list__)
    return new_list


def get_residuals_new(interval_d, drift_params):
    new_list = []
    for list_num, list_ in enumerate(interval_d):
        new_list__ = []
        for num, drift_param in enumerate(drift_params[list_num]):
            for num_, interval in enumerate(list_, start=0):
                new_list__.append([interval[0] - (num_ + 1) * drift_param[0][1] - drift_param[1][1],
                                   interval[1] - (num_ + 1) * drift_param[0][0] - drift_param[1][0]])
        new_list.append(new_list__)
    return new_list


def read_intervals_data(files_with_data):
    data_rows = []
    eps = 1e-04

    for file in files_with_data:
        data_r = genfromtxt(file, delimiter=';')
        data_rows.append([[val[0] - eps, val[0] + eps] for val in data_r][1:201])
    return data_rows


def draw_residuals(residuals_l, intersection_, title='', save_path=None):
    data_len = (len(residuals_l))
    x = np.arange(1, data_len + 1, 1, dtype=int)
    y_err = [get_radius(interval) for interval in residuals_l]
    y = [get_middle(interval) for interval in residuals_l]
    plt.errorbar(x, y, yerr=y_err, ecolor='cornflowerblue', label=f'residuals', elinewidth=0.8, capsize=4,
                 capthick=1)
    x1, y1 = [1, 200], [intersection_[0], intersection_[0]]
    x2, y2 = [1, 200], [intersection_[1], intersection_[1]]
    x3, y3 = [1, 200], [(intersection_[1] + intersection_[0]) / 2, (intersection_[1] + intersection_[0]) / 2]
    plt.plot(x1, y1, 'r--', label=f'[{y1[0]:.2e}, {y2[1]:.2e}]')
    plt.plot(x2, y2, 'r--')
    plt.plot(x3, y3, 'k--', label=f'mid={y3[0]:.2e}')
    plt.legend(frameon=False)
    plt.title(title)
    plt.yticks(fontsize=8)
    plt.xlabel("n")
    plt.ylabel("residual")
    if save_path is not None:
        plt.savefig(f'{save_path}/regression_intervals.png')
    #plt.show()


def intervals_regularization(residuals, edge_point, intersection_, need_visualize=False, title='', label=''):
    w_list = [1] * len(residuals)
    left_intervals = residuals[:edge_point].copy()
    for num, interval in enumerate(left_intervals):
        mid = (left_intervals[num][1] + left_intervals[num][0]) / 2
        if interval[0] > intersection_[0]:
            left_intervals[num][0] = intersection_[0]
            left_intervals[num][1] = mid + (mid - intersection_[0])
            w_list[num] = (mid - intersection_[0]) / ((interval[1] - interval[0]) / 2)
        if interval[1] < intersection_[1]:
            left_intervals[num][1] = intersection_[1]
            left_intervals[num][0] = mid - (intersection_[1] - mid)
            w_list[num] = (intersection_[1] - mid) / ((interval[1] - interval[0]) / 2)
    if need_visualize:
        plt.hist(w_list, label=label)
        plt.xlabel(label)
        plt.legend(frameon=False)
        plt.title(title)
        plt.show()
    return left_intervals + residuals[edge_point:]


def draw_parameters(params_1, params_2=None, title='', param_name='', param_name_2=''):
    x = np.arange(1, len(params_1) + 1, 1, dtype=int)
    fig, ax = plt.subplots()
    if params_2:
        ax.plot(x, np.fabs(np.array(params_1)), label=param_name)
        ax.plot(x, 1 - np.array(params_2), label=param_name_2)
    else:
        ax.plot(x, params_1, label=param_name)
    plt.xlabel("n")
    plt.ylabel(param_name)
    ax.legend()
    ax.set_title(title)
    plt.show()


def get_matlab_form(intervals):
    res_str = '['
    for interval in intervals:
        res_str = f'{res_str}[{interval[0]}, {interval[1]}];'
    print(f'{res_str[:-1]}]')


def draw_mode(mode_data, title='Mode'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_len = len(mode_data)
    for num, mode_data_inst in enumerate(mode_data):
        x1, y1 = [mode_data_inst[0][0], mode_data_inst[0][1]], [mode_data_inst[1], mode_data_inst[1]]
        ax.plot(x1, y1, 'r')
        if num < data_len - 1:
            x2, y2 = [mode_data_inst[0][1], mode_data_inst[0][1]], [mode_data_inst[1], mode_data[num + 1][1]]
            ax.plot(x2, y2, 'r')
    ax.set_xlabel('ticks')
    ax.set_ylabel('frequency')
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":

    data_postfix = '700nm_0.03.csv'
    interval_data = read_intervals_data(
        [f'Канал 1_{data_postfix}', f'Канал 2_{data_postfix}'])

    edge_points_ = [[38, 181], [59, 132]]
    # здесь свои параметры регрессии
    intervals_regression_params_3 = [[([6.1838e-05, 1.8761e-04], [4.7913e-02, 4.8751e-02]),
                                      ([4.9625e-05, 5.8875e-05], [4.9238e-02, 4.9505e-02]),
                                      ([4.8173e-05, 5.2607e-05], [4.9063e-02, 4.9538e-02]),
                                      ([5.6000e-05, 5.6250e-05], [4.8074e-02, 4.8120e-02]),
                                      ([1.4967e-04, 2.0900e-04], [1.8302e-02, 3.0040e-02])],
                                     [([6.4333e-05, 9.3000e-05], [5.3075e-02, 5.3222e-02]),
                                      ([3.7922e-05, 3.8386e-05], [5.3563e-02, 5.3579e-02]),
                                      ([5.4972e-05, 6.5068e-05], [5.1832e-02, 5.2928e-02]),
                                      ([3.3118e-05, 3.7682e-05], [5.5476e-02, 5.6228e-02]),
                                      ([8.6000e-05, 1.0790e-04], [4.2296e-02, 4.6523e-02])]]

    intervals_residuals = get_residuals(interval_data, edge_points_, intervals_regression_params_3)
    intervals_residuals_1 = get_residuals_new(interval_data, [[intervals_regression_params_3[0][1]],
                                                              [intervals_regression_params_3[1][1]]])

#    LetsDraw0 = False
#    if LetsDraw0:
#        for num_, res_list in enumerate(interval_data):
#            with open(f'data{num_ + 1}.mat', 'w') as f:
#                for elem in res_list:
#                    line = str(elem[0]) + ' ' + str(elem[1]) + '\n'
#                    f.write(line)
#            with open(f'result{num_ + 1}.txt', 'r') as r:
#                lines = r.read()
#                lst = list(ast.literal_eval(lines))
#            draw_mode(lst)
#            plt.show()

    LetsGraw1 = True
    if LetsGraw1:
        for num_, res_list in enumerate(intervals_residuals_1):
            inter_w = get_intersections_wrong_interval(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            inters = get_intersections(intervals_residuals_1[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            new_res_list = intervals_regularization(res_list, edge_points_[num_][0], inters, True,
                                                    f'Weight of regularization, ch_{num_ + 1}', f'w Channel {num_ + 1}')
            infls, intersection = get_influences(res_list, inters)
            m_l = max([res[0] for res in infls])
            fig_, ax_ = draw_data_status_lawn([0, max(m_l, 2)], title=f'Influences with central regression, Channel {num_ + 1}')
            for infl in infls:
                add_point(infl, ax_)
            fig_.show()
            plt.show()
            draw_residuals(res_list, intersection, title=f'Residuals with central regression, Channel {num_ + 1}')
#            with open(f'data{num_ + 1}.mat', 'w') as f:
#                for elem in res_list:
#                    line = str(elem[0]) + ' ' + str(elem[1]) + '\n'
#                    f.write(line)
#            with open(f'result{num_ + 1}.txt', 'r') as r:
#                lines = r.read()
#                lst = list(ast.literal_eval(lines))
#            draw_mode(lst)
#            plt.show()


    LetsGraw2 = False
    if LetsGraw2:
        for num_, res_list in enumerate(intervals_residuals):
            inter_w = get_intersections_wrong_interval(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            print(inter_w)
            inters = get_intersections(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            infls, intersection = get_influences(res_list, inters)
            m_l = max([res[0] for res in infls])
            fig_1, ax_1 = draw_data_status_lawn([0, max(m_l, 2)], title=f'Influences with central regression, Channel {num_ + 1}')
            for infl in infls:
                add_point(infl, ax_1)
            fig_1.show()
            plt.show()
            draw_residuals(res_list, intersection, title=f'Residuals with central regression, Channel {num_ + 1}')
#            with open(f'data{num_ + 1}.mat', 'w') as f:
#                for elem in res_list:
#                    line = str(elem[0]) + ' ' + str(elem[1]) + '\n'
#                    f.write(line)
#            with open(f'result{num_ + 1}.txt', 'r') as r:
#                lines = r.read()
#                lst = list(ast.literal_eval(lines))
#            draw_mode(lst)
#            plt.show()


    intervals_residuals = get_residuals_new(interval_data, [[intervals_regression_params_3[0][1]],
                                                            [intervals_regression_params_3[1][1]]])

    LetsGraw3 = False
    if LetsGraw3:
        for num_, res_list in enumerate(intervals_residuals):
            inter_w = get_intersections_wrong_interval(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            print(inter_w)
            inters = get_intersections(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            infls, intersection = get_influences(res_list, inters)
            m_l = max([res[0] for res in infls])
            fig_, ax_ = draw_data_status_lawn([0, max(m_l, 2)], title=f'Influences with all regressions, Channel {num_ + 1}')
            for infl in infls:
                add_point(infl, ax_)
            fig_.show()
            plt.show()
            draw_residuals(res_list, intersection, title=f'Residuals with all regressions, Channel {num_ + 1}')
#            with open(f'data{num_ + 1}.mat', 'w') as f:
#                for elem in res_list:
#                    line = str(elem[0]) + ' ' + str(elem[1]) + '\n'
#                    f.write(line)
#            with open(f'result{num_ + 1}.txt', 'r') as r:
#                lines = r.read()
#                lst = list(ast.literal_eval(lines))
#            draw_mode(lst)
#            plt.show()

    LetsGraw4 = False
    if LetsGraw4:
        for num_, res_list in enumerate(intervals_residuals):
            inter_w = get_intersections_wrong_interval(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            inters = get_intersections(intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            new_res_list = intervals_regularization(res_list, edge_points_[num_][0], inters, True,
                                          f'Weight of regularization, ch_{num_ + 1}', f'w Channel {num_ + 1}')
            infls, intersection = get_influences(new_res_list, inters)
            draw_residuals(new_res_list, intersection, title=f'Residuals with central regression, Channel {num_ + 1}')
            m_l = max([res[0] for res in infls])
#            draw_parameters([infl[0] for infl in infls], title=f'High leverage, Channel {num_ + 1}', param_name='l')
#            draw_parameters([infl[1] for infl in infls], title=f'Relative residual, Channel {num_ + 1}', param_name='|r|',
#                            params_2=[infl[0] for infl in infls], param_name_2='1-l')
            fig_, ax_ = draw_data_status_lawn([0, max(m_l, 2)], title=f'Influences with central regression, Channel {num_ + 1}')
            for infl in infls:
                add_point(infl, ax_)
            fig_.show()
            plt.show()

            # for num_, res_list in enumerate(intervals_residuals_1):
            #     inter_w = get_intersections_wrong_interval(
            #         intervals_residuals[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            #     inters = get_intersections(intervals_residuals_1[num_][edge_points_[num_][0]:edge_points_[num_][1]])
            #     infls, intersection = get_influences(res_list, inters)
            #     m_l = max([res[0] for res in infls])
            #     fig_, ax_ = draw_data_status_lawn([0, max(m_l, 2)],
            #                                       title=f'Influences with central regression, Channel {num_ + 1}')
            #     for infl in infls:
            #         add_point(infl, ax_)
            #     fig_.show()
            #     plt.show()
            #     draw_residuals(res_list, intersection, title=f'Residuals with central regression, Channel {num_ + 1}')

#            with open(f'data{num_ + 1}.mat', 'w') as f:
#                for elem in new_res_list:
#                    line = str(elem[0]) + ' ' + str(elem[1]) + '\n'
#                    f.write(line)
#            with open(f'result{num_ + 1}.txt', 'r') as r:
#                lines = r.read()
#                lst = list(ast.literal_eval(lines))
#            draw_mode(lst)
#            plt.show()
