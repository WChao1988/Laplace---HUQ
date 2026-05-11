import numpy as np
import pandas as pd
import pickle
import os
import torch
from utls.mat_interpolator import MatInterpolator
from utls.longwave import LongWave

dir = os.path.dirname(__file__)
order = 9


class ProbablisticLoss(torch.nn.Module):
    def __init__(self):
        super(ProbablisticLoss, self).__init__()

    def forward(self, phi, ship, labda):
        return torch.mean(torch.log(labda + 1e-6) + torch.abs(phi - ship) / (labda + 1e-6))


class GGMModel:
    def __init__(self, delta_rho, lat_up, lat_down, lon_left, lon_right, radius=0.5, sigma=0.5):
        self.delta_rho = delta_rho
        self.lat_up, self.lat_down = lat_up, lat_down
        self.lon_left, self.lon_right = lon_left, lon_right
        self.vgg_matrix = None
        self.ship_grid = None
        self.reference_depth = None
        self.G = 6.67e-3
        self.radius, self.sigma = radius, sigma
        self.gravity_long_model = None
        self.gravity_model = None

    def __calculate_short_gravity__(self, depth_matrix):
        return 2 * np.pi * self.G * self.delta_rho * (depth_matrix - self.reference_depth)

    def fit(self, vgg_matrix, ship_grid, reference_depth):
        self.vgg_matrix = vgg_matrix
        self.ship_grid = ship_grid
        self.reference_depth = reference_depth
        mask = ship_grid != 0
        vgg_short_ship_series = self.__calculate_short_gravity__(ship_grid[mask])
        vgg_long_ship = np.zeros_like(vgg_matrix)
        vgg_long_ship[mask] = vgg_matrix[mask] - vgg_short_ship_series
        self.gravity_long_model = MatInterpolator(radius=self.radius, sigma=self.sigma)
        self.gravity_long_model.fit(vgg_long_ship, self.lat_up, self.lat_down, self.lon_left, self.lon_right)
        self.gravity_model = MatInterpolator()
        self.gravity_model.fit(vgg_matrix, self.lat_up, self.lat_down, self.lon_left, self.lon_right)

    def __calculate_ship_depth__(self, short_gravity_series):
        return short_gravity_series / (2 * np.pi * self.G * self.delta_rho) + self.reference_depth

    def predict(self, target_df):
        vgg_df_target = self.gravity_model.predict_by_nearest(target_df)
        vgg_long_target = self.gravity_long_model.predict_by_inverse_distance(target_df)
        short_gravity_series = vgg_df_target['bathymetry'] - vgg_long_target['bathymetry']
        result = self.__calculate_ship_depth__(short_gravity_series)
        output = target_df.copy()
        output['bathymetry'] = result
        return output

    def prediction_matrix(self):
        mask = self.ship_grid != 0
        vgg_short_ship_series = self.__calculate_short_gravity__(self.ship_grid[mask])
        vgg_long_ship = np.zeros_like(self.vgg_matrix)
        vgg_long_ship[mask] = self.vgg_matrix[mask] - vgg_short_ship_series
        vgg_long_model = LongWave(vgg_long_ship, self.lat_up,
                                  self.lat_down, self.lon_left, self.lon_right, self.radius, self.sigma)
        vgg_long_matrix = vgg_long_model.long_wave_matrix()
        vgg_short_matrix = self.vgg_matrix - vgg_long_matrix
        result = self.__calculate_ship_depth__(vgg_short_matrix)
        return result


if __name__ == "__main__":
    # ------------------------------------------------ 加载数据
    data = pickle.load(open(os.path.join(dir, 'data_seamount', f'data_{order}.pkl'), 'rb'))
    vgg_matrix = data['vgg_matrix']
    vgg_df = data['vgg_df']
    ship_grid_train = data['ship_grid_train']
    ship_grid_test = data['ship_grid_test']
    topography_matrix = data['topography_matrix']
    tri_matrix = data['tri_matrix']
    mask_train = data['mask_train']
    mask_test = data['mask_test']
    target_size = data['target_size']
    lat_up, lat_down, lon_left, lon_right = 21, 16, 149, 154

    # ------------------------------------------------ GGM 训练
    delta_rho_best = 1.63
    rmse_best = np.inf
    reference_depth = ship_grid_train[mask_train].mean()
    ship_train = ship_grid_train.copy()
    radius = 0.8
    sigma = 0.0006

    for delta_rho in np.arange(0.9, 1.8, 0.01):
        gm = GGMModel(delta_rho, lat_up, lat_down, lon_left, lon_right, radius=radius, sigma=sigma)
        gm.fit(vgg_matrix, ship_train, reference_depth=reference_depth)
        complete_matrix = gm.prediction_matrix()
        difference = complete_matrix[mask_train] - ship_train[mask_train]
        rmse = np.sqrt(((difference) ** 2).mean())
        if rmse < rmse_best:
            rmse_best = rmse
            delta_rho_best = delta_rho
            print(delta_rho_best, rmse_best)

    # ------------------------------------------------ GGM 估计
    gm = GGMModel(delta_rho_best, lat_up, lat_down, lon_left, lon_right, radius=radius, sigma=sigma)
    gm.fit(vgg_matrix, ship_train, reference_depth=reference_depth)
    complete_df = gm.predict(vgg_df)
    complete_matrix = gm.prediction_matrix()
    difference_train = complete_matrix[mask_train] - ship_train[mask_train]
    difference_test = complete_matrix[mask_test] - ship_grid_test[mask_test]
    rmse_train = np.sqrt(((difference_train) ** 2).mean())
    rmse_test = np.sqrt((difference_test ** 2).mean())
    print(rmse_train)
    print(rmse_test)

    # ------------------------------------------------ 构建 tensor 特征
    topography_tensor_train = torch.tensor(topography_matrix[mask_train], dtype=torch.float32)
    topography_tensor_test = torch.tensor(topography_matrix[mask_test], dtype=torch.float32)
    tri_tensor_train = torch.tensor(tri_matrix[mask_train], dtype=torch.float32)
    tri_tensor_test = torch.tensor(tri_matrix[mask_test], dtype=torch.float32)
    ship_tensor_train = torch.tensor(ship_grid_train[mask_train], dtype=torch.float32)
    phi_tensor_train = torch.tensor(complete_matrix[mask_train], dtype=torch.float32)

    # ------------------------------------------------ 训练不确定性模型
    model = torch.nn.Sequential(torch.nn.Linear(in_features=2, out_features=8, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Linear(8, 8),
                                torch.nn.ReLU(),
                                torch.nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ProbablisticLoss()
    input_train = torch.stack([topography_tensor_train, tri_tensor_train], dim=1)
    nround = 20000
    loss_history = []
    for epoch in range(nround):
        optimizer.zero_grad()
        output = torch.nn.functional.softplus(model(input_train).squeeze())
        loss = criterion(phi=phi_tensor_train, ship=ship_tensor_train, labda=output)
        loss.backward()
        optimizer.step()
        print(f'epoch={epoch}, loss={loss.item():.4f}')
        if epoch > 100 and epoch % 100 == 0:
            loss_history.append(loss.item())

    # ------------------------------------------------ 保存模型与结果
    name = f"model_{order}"
    pickle.dump(loss_history, open(dir + f'/results1_16-21/loss_history_{name}.lst', 'wb'))
    pickle.dump(model, open(dir + f'/results1_16-21/{name}.model', 'wb'))

    model_x = pickle.load(open(dir + f'/results1_16-21/{name}.model', 'rb'))
    labda_tensor_train = 1 / torch.nn.functional.softplus(model_x(input_train).squeeze())

    input_test = torch.stack([topography_tensor_test, tri_tensor_test], dim=1)
    labda_tensor_test = 1 / torch.nn.functional.softplus(model(input_test).squeeze())
    ship_tensor_test = torch.tensor(ship_grid_test[mask_test], dtype=torch.float32)
    phi_tensor_test = torch.tensor(complete_matrix[mask_test], dtype=torch.float32)

    # ------------------------------------------------ 保存推理结果
    results = {
        'complete_matrix': complete_matrix,
        'phi_tensor_train': phi_tensor_train,
        'phi_tensor_test': phi_tensor_test,
        'ship_tensor_train': ship_tensor_train,
        'ship_tensor_test': ship_tensor_test,
        'labda_tensor_train': labda_tensor_train,
        'labda_tensor_test': labda_tensor_test,
        'topography_tensor_train': topography_tensor_train,
        'topography_tensor_test': topography_tensor_test,
        'tri_tensor_train': tri_tensor_train,
        'tri_tensor_test': tri_tensor_test,
        'loss_history': loss_history,
    }
    pickle.dump(results, open(dir + f'/results1_16-21/results_{order}.pkl', 'wb'))
    print("Results saved.")