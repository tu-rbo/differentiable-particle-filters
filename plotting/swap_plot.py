import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import os

results = None

# matplotlib.rcParams.update({'font.size': 12})

color_list = plt.cm.tab10(np.linspace(0, 1, 10))
colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3]}
labels = {'lstm': 'LSTM', 'pf_e2e': 'DPF (e2e)', 'pf_ind_e2e': 'DPF (ind+e2e)', 'pf_ind': 'DPF (ind)', 'ff': 'FF', 'odom': 'Odom. baseline'}
# conditions = ['normal', 'no_motion_likelihood', 'learn_odom', 'no_proposer']
# conditions = ['normal', 'learn_odom', 'no_inject']
# clabels = {'normal': 'Default', 'no_motion_likelihood': 'W/o motion likelihood', 'learn_odom': 'Learned odometry', 'no_proposer': 'W/o particle proposer', 'no_inject': "No inject"}

if False:
    test_conditions = ['odom5_imgTG', 'odom10_imgTG']
    
    conditions = ['orig_odom5_imgTG', 'odom5_imgTG_odom5_imgTG',
                  'odom10_imgTG_odom5_imgTG', 'odom5_imgTG_odom10_imgTG',
                    'odom10_imgTG_odom10_imgTG', 'orig_odom10_imgTG']

    clabels = {'orig_odom5_imgTG':'5',
               'odom5_imgTG_odom5_imgTG':'mo5,me5',
               'odom5_imgTG_odom10_imgTG':'mo5,me10',
               'orig_odom10_imgTG':'10',
               'odom10_imgTG_odom10_imgTG': 'mo10,me10',
               'odom10_imgTG_odom5_imgTG': 'mo10,me5',
               'odom5_imgTG': '5',
               'odom10_imgTG': '10',
                }
    xlabels = ['A', 'B']
    exp = 'swapmo'
else:

    test_conditions = ['odom10_imgG', 'odom10_imgTG']

    conditions = ['orig_odom10_imgG', 'odom10_imgG_odom10_imgG', 'odom10_imgG_odom10_imgTG',
                  'odom10_imgTG_odom10_imgG', 'odom10_imgTG_odom10_imgTG', 'orig_odom10_imgTG']

    # clabels = {'orig_odom10_imgG':'G',
    #            'odom10_imgG_odom10_imgG':'meG,moG',
    #            'odom10_imgG_odom10_imgTG':'meG,moTG',
    #            'orig_odom10_imgTG':'TG',
    #            'odom10_imgTG_odom10_imgTG': 'meTG,moTG',
    #            'odom10_imgTG_odom10_imgG': 'meTG,moG',
    #            'odom10_imgG': 'G',
    #            'odom10_imgTG': 'TG',
    #             }

    clabels = {'orig_odom10_imgG':'A(A)*',
           'odom10_imgG_odom10_imgG':'A(A)',
           'odom10_imgG_odom10_imgTG':'A(B)',
           'orig_odom10_imgTG':'B(B)*',
           'odom10_imgTG_odom10_imgTG': 'B(B)',
           'odom10_imgTG_odom10_imgG': 'B(A)',
           'odom10_imgG': 'A',
           'odom10_imgTG': 'B',
            }
    xlabels = ['C', 'D']
    exp = 'swapme'

vmax = 0.4

task = 'nav02'
# methods = ['pf_ind', 'pf_e2e', 'pf_ind_e2e', 'lstm']
methods = ['pf_ind_e2e']

# load results
results = dict()

count = 0
for cond in conditions:
    # log_path = '/home/rbo/Desktop/log/'+task+'_ab1'
    log_path = '../log/'+exp+'/'+cond
    for filename in [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]:
        full_filename = os.path.join(log_path, filename)
        print('loading {}:'.format(count) + full_filename + ' ...')
        try:
            # if 'DeepThought' not in filename:
            # if 'DeepThought' in filename:
            with open(full_filename, 'rb') as f:
                result = pickle.load(f)
                # result_name = result['task'][0] + '/' + result['method'][0] + '/' + str(result['num_episodes'][0]) + '/' + result['condition'][0]
                result_name = cond #+ '_' + result['exp_params'][0]['file_ending'] #result['exp_params'][0]['task'] + '/' + result['exp_params'][0]['method'] + '/' + str(result['exp_params'][0]['num_episodes']) + '/' + result['exp_params'][0]['ab_cond']
                print(result_name)
                if result_name not in results.keys():
                    results[result_name] = result
                else:
                    for key in result.keys():
                        if key in results[result_name].keys():
                            results[result_name][key] += result[key]
                        else:
                            results[result_name][key] = result[key]
                        # print(result_name, key)
                count += 1
        except Exception as e:
            print(e)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

print()
for result_name, r in results.items():
    print(result_name, len(r['exp_params']))
    # print(result_name, len(r['test_odom5_imgTG_mse']))

print('Loaded {} results'.format(count))


task = 'nav02'
step = 3

episodes = [1000]
# episodes = [1000]


means = []
ses = []

for c, condition in enumerate(conditions):

    means.append(np.zeros([len(test_conditions), 5]))
    ses.append(np.zeros([len(test_conditions), 5]))

    for tc, test_condition in enumerate(test_conditions):

        result_name = condition
        if result_name in results.keys():
            result = results[result_name]

            hist = np.array([[h[i] for i in range(0, 50, 10)] for h in result['test_'+test_condition+'_hist' ]])  # result x time x sqe [.0, 0.1, .., 10.0]
            err = 1. - np.sum(hist[:,:,:10], axis=-1) # sqe < 1.0
            # err = np.sum(hist[:,:,:10], axis=-1) # sqe < 1.0
            print(result_name, err)
            means[c][tc] = np.mean(err, axis=0)
            ses[c][tc] = np.std(err, axis=0, ddof=1) / np.sqrt(len(err))

        else:
            print(result_name + 'not found')
            means[tc] *= np.nan
            ses[tc] *= np.nan


# means[num_episodes]['min'] = np.stack([means[num_episodes][method] for method in methods], axis=0).min(axis=1)

# ax = fig.add_subplot(111)
# # Turn off axis lines and ticks of the big subplot
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

m = np.array(means)[:, :, step].T
s = np.array(ses)[:, :, step].T

for i in range(2):
    z = m[i, :]
    y = s[i, :]
    x = [[z[0]] * 2 + [z[2]] * 2,
         [z[1]] * 2 + [z[2]] * 2,
         [z[3]] * 2 + [z[5]] * 2,
         [z[3]] * 2 + [z[4]] * 2]
    plt.figure(i, [1.35,1.35])
    plt.imshow(x, interpolation='nearest', vmin=-0.33*vmax, vmax=vmax, cmap='viridis_r')
    plt.plot([-0.5, 3.5],[1.5, 1.5], '-w', linewidth=0.5)
    plt.plot([1.5, 1.5],[-0.5, 3.5], '-w', linewidth=0.5)

    for j, x_coord, y_coord, value, s_value in [
            (0, 0.5, 0, z[0], y[0]),
            (1, 0.5, 1, z[1], y[1]),
            (2, 2.5, 0.5, z[2], y[2]),
            (3, 0.5, 2.5, z[3], y[3]),
            (5, 2.5, 2, z[5], y[5]),
            (4, 2.5, 3, z[4], y[4])]:
        if j == 0 or j == 5:
            # text = '{:.4s}*\n+-{:.4s}'.format('{:.3f}'.format(value)[1:],'{:.2f}'.format(s_value)[1:])
            text = ' {:.4s}*'.format('{:.3f}'.format(value)[1:],'{:.2f}'.format(s_value)[1:])
        else:
            # text = '{:.4s}\n+-{:.4s}'.format('{:.3f}'.format(value)[1:],'{:.2f}'.format(s_value)[1:])
            text = '{:.4s}'.format('{:.3f}'.format(value)[1:],'{:.2f}'.format(s_value)[1:])

        plt.text(x_coord, y_coord, text, va='center', ha='center', color='white', fontweight='normal')

    plt.gca().set_aspect('equal')

    plt.xlabel('Motion model')
    plt.xticks([0.5, 2.5], xlabels)
    plt.ylabel('Measurem. model')
    plt.yticks([0.5, 2.5], xlabels)
    plt.tight_layout(0.0, 0.0, 0.0)
    print('saving')
    plt.savefig('../plots/cr/'+exp+'%s.pdf' % i, bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)

plt.figure('colorbar', [0.6, 1.35])
a = np.array([[0.0, 0.3]])
img = plt.imshow(a, cmap="viridis_r", vmin=-0.33*vmax, vmax=vmax)
plt.gca().set_visible(False)
cax = plt.axes([0.0, 0.2, 0.1, 0.65])
plt.colorbar(orientation="vertical", cax=cax, label='Error rate', boundaries=np.linspace(0,0.4,100), ticks=np.linspace(0.0, 0.4, 5))

plt.savefig('../plots/cr/colorbar.pdf'.format(s), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)


plt.show()

