import numpy as np

from utils.data_utils_kitti import wrap_angle

class OdometryBaseline():

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def predict(self, sess, batch, **kwargs):
        seq_len = batch['s'].shape[1]

        prediction = np.zeros_like(batch['s'])
        state = batch['s'][:, 0, :]
        # print('shape:', batch['s'].shape)
        prediction[:, 0, :] = state
        for i in range(1, seq_len):

            action = batch['a'][:, i, :]
            theta = state[:, 2:3]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            new_x = state[:, 0:1] + (action[:, 0:1] * cos_theta + action[:, 1:2] * sin_theta)
            new_y = state[:, 1:2] + (action[:, 0:1] * sin_theta - action[:, 1:2] * cos_theta)
            new_theta = wrap_angle(state[:, 2:3] + action[:, 2:3])
            # copy old and set new particles
            state = np.concatenate([new_x, new_y, new_theta], axis=-1)
            prediction[:, i, :] = state
        return prediction

    def predict_kitti(self, sess, batch, **kwargs):
        seq_len = batch['s'].shape[1]

        prediction = np.zeros_like(batch['s'])
        state = batch['s'][:, 0, :]
        # print('shape:', batch['s'].shape)
        prediction[:, 0, :] = state
        for i in range(1, seq_len):

            time = 0.103

            action = batch['a'][:, i, :]
            heading = state[:, 2:3]
            wrap_angle(heading)
            sin_heading = np.sin(heading)
            cos_heading = np.cos(heading)

            # ang_acc = (noisy_actions[:, :, 1:2] * noisy_actions[:, :, 2:3])/(noisy_actions[:, :, 0:1] ** 2)

            acc_north = action[:, 0:1] * sin_heading + action[:, 1:2] * cos_heading
            acc_east = - action[:, 1:2] * sin_heading + action[:, 0:1] * cos_heading

            new_north = state[:, 0:1] + state[:, 3:4] * time
            new_east = state[:, 1:2] + state[:, 4:5] * time
            new_theta = state[:, 2:3] + state[:, 5:6] * time
            wrap_angle(new_theta)
            new_vn = state[:, 3:4] + acc_north * time
            new_ve = state[:, 4:5] + acc_east * time
            new_theta_dot = state[:, 5:6] + action[:, 2:3] * time

            state = np.concatenate([new_north, new_east, new_theta, new_vn, new_ve, new_theta_dot], axis=-1)
            prediction[:, i, :] = state
        return prediction
