import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pickle
import os
from scipy.stats import expon
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

dir = os.path.dirname(__file__)
order = 9

# ------------------------------------------------ 加载数据与结果
data = pickle.load(open(os.path.join(dir, 'results1_16-21', f'data_{order}.pkl'), 'rb'))
results = pickle.load(open(os.path.join(dir, 'results1_16-21', f'results_{order}.pkl'), 'rb'))

mask_train = data['mask_train']
mask_test = data['mask_test']
target_size = data['target_size']
ship_grid_test = data['ship_grid_test']
complete_matrix = results['complete_matrix']
phi_tensor_train = results['phi_tensor_train']
phi_tensor_test = results['phi_tensor_test']
ship_tensor_train = results['ship_tensor_train']
ship_tensor_test = results['ship_tensor_test']
labda_tensor_train = results['labda_tensor_train']
labda_tensor_test = results['labda_tensor_test']
topography_tensor_train = results['topography_tensor_train']
tri_tensor_train = results['tri_tensor_train']
loss_history = results['loss_history']

plot_dir = f'plot_{order}'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# ------------------------------------------------ GGM scatter
plt.figure()
plt.scatter(ship_grid_test[mask_test], complete_matrix[mask_test], s=0.5)
plt.plot([ship_grid_test[mask_test].min(), ship_grid_test[mask_test].max()],
         [ship_grid_test[mask_test].min(), ship_grid_test[mask_test].max()], 'r--', lw=2)
plt.savefig(f'plot_{order}/scatter_plot_GGM_test_{order}.png', dpi=300)
plt.show()

# ------------------------------------------------ 1. 训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', linewidth=1.5)
plt.title('Training Loss Curve')
plt.xlabel('Epoch (every 100 steps)')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.savefig(f'plot_{order}/training_loss_{order}.png', dpi=300)
plt.show()


# ------------------------------------------------ 2. 残差分析
def plot_residual_analysis():
    """残差分布和模型假设验证"""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    train_res = (phi_tensor_train - ship_tensor_train).abs().detach().numpy()
    test_res = (phi_tensor_test - ship_tensor_test).abs().detach().numpy()
    train_lambda = labda_tensor_train.detach().numpy()
    test_lambda = labda_tensor_test.detach().numpy()

    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(train_res, bins=50, kde=True, color='blue', alpha=0.5, label='Train', ax=ax1, stat='density')
    sns.histplot(test_res, bins=50, kde=True, color='red', alpha=0.5, label='Test', ax=ax1, stat='density')
    ax1.set_title('Absolute Residual Distribution')
    ax1.set_xlabel('|φ - ship|')
    ax1.set_ylabel('Density')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    param_train = expon.fit(train_res)
    x_train = np.linspace(0, np.max(train_res), 100)
    pdf_train = expon.pdf(x_train, *param_train)
    param_test = expon.fit(test_res)
    x_test = np.linspace(0, np.max(test_res), 100)
    pdf_test = expon.pdf(x_test, *param_test)
    sns.histplot(train_res, stat='density', bins=50, color='blue', alpha=0.3, label='Train Data', ax=ax2)
    ax2.plot(x_train, pdf_train, 'b-', lw=2, label='Train Exponential Fit')
    sns.histplot(test_res, stat='density', bins=50, color='red', alpha=0.3, label='Test Data', ax=ax2)
    ax2.plot(x_test, pdf_test, 'r-', lw=2, label='Test Exponential Fit')
    ax2.set_title('Residual Distribution vs Exponential Fit')
    ax2.set_xlabel('|φ - ship|')
    ax2.set_ylabel('Density')
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(1 / train_lambda, train_res, alpha=0.3, color='blue', label='Train')
    ax3.scatter(1 / test_lambda, test_res, alpha=0.3, color='red', label='Test')
    max_val = max(np.max(1 / train_lambda), np.max(1 / test_lambda))
    ax3.plot([0, max_val], [0, max_val], 'k--', label='Theoretical: |residual| = 1/λ')
    ax3.set_title('Theoretical Relationship: |Residual| vs 1/λ')
    ax3.set_xlabel('1/λ')
    ax3.set_ylabel('|φ - ship|')
    ax3.legend()

    train_ci = 3 / train_lambda
    test_ci = 3 / test_lambda
    train_coverage = np.mean(train_res <= train_ci) * 100
    test_coverage = np.mean(test_res <= test_ci) * 100

    ax4 = fig.add_subplot(gs[1, 1])
    categories = ['Train Coverage', 'Test Coverage', 'Theoretical (95%)']
    values = [train_coverage, test_coverage, 95]
    colors = ['blue', 'red', 'green']
    bars = ax4.bar(categories, values, color=colors)
    ax4.set_ylim(0, 100)
    ax4.set_title('95% Confidence Interval Coverage')
    ax4.set_ylabel('Coverage (%)')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'plot_{order}/residual_analysis_{order}.png', dpi=300)
    plt.show()
    return train_coverage, test_coverage


train_cov, test_cov = plot_residual_analysis()
print(f"Train Coverage: {train_cov:.2f}%, Test Coverage: {test_cov:.2f}%")


# ------------------------------------------------ 3. 预测性能评估
def plot_performance_evaluation():
    """预测水深与实际水深比较"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    train_phi = phi_tensor_train.detach().numpy()
    train_ship = ship_tensor_train.detach().numpy()
    test_phi = phi_tensor_test.detach().numpy()
    test_ship = ship_tensor_test.detach().numpy()

    train_r2 = r2_score(train_ship, train_phi)
    test_r2 = r2_score(test_ship, test_phi)
    train_mae = mean_absolute_error(train_ship, train_phi)
    test_mae = mean_absolute_error(test_ship, test_phi)

    axs[0].scatter(train_phi, train_ship, alpha=0.5, color='blue')
    min_val = min(np.min(train_phi), np.min(train_ship))
    max_val = max(np.max(train_phi), np.max(train_ship))
    axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axs[0].set_title(f'Train Set: φ vs ship (R²={train_r2:.3f}, MAE={train_mae:.2f})')
    axs[0].set_xlabel('Predicted Depth (φ)')
    axs[0].set_ylabel('Actual Depth (ship)')
    axs[0].grid(True)

    axs[1].scatter(test_phi, test_ship, alpha=0.5, color='red')
    min_val = min(np.min(test_phi), np.min(test_ship))
    max_val = max(np.max(test_phi), np.max(test_ship))
    axs[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axs[1].set_title(f'Test Set: φ vs ship (R²={test_r2:.3f}, MAE={test_mae:.2f})')
    axs[1].set_xlabel('Predicted Depth (φ)')
    axs[1].set_ylabel('Actual Depth (ship)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'plot_{order}/performance_evaluation_{order}.png', dpi=300)
    plt.show()


plot_performance_evaluation()


# ------------------------------------------------ 4. λ 分析
def plot_lambda_analysis():
    """λ值分布及其与地形特征的关系"""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    train_lambda = labda_tensor_train.detach().numpy()
    test_lambda = labda_tensor_test.detach().numpy()
    train_topo = topography_tensor_train.detach().numpy()
    train_tri = tri_tensor_train.detach().numpy()

    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(train_lambda, bins=50, kde=True, color='blue', alpha=0.5, label='Train', ax=ax1, stat='density')
    sns.histplot(test_lambda, bins=50, kde=True, color='red', alpha=0.5, label='Test', ax=ax1, stat='density')
    ax1.set_title('λ Value Distribution')
    ax1.set_xlabel('λ')
    ax1.set_ylabel('Density')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(train_topo, train_lambda, c=train_tri,
                     cmap='viridis', alpha=0.6, vmin=np.percentile(train_tri, 5),
                     vmax=np.percentile(train_tri, 95))
    ax2.set_title('λ vs Topography (Color: TRI)')
    ax2.set_xlabel('Topography (Depth)')
    ax2.set_ylabel('λ')
    fig.colorbar(sc, ax=ax2, label='TRI')

    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(train_tri, train_lambda, c=train_topo,
                     cmap='plasma', alpha=0.6, vmin=np.percentile(train_topo, 5),
                     vmax=np.percentile(train_topo, 95))
    ax3.set_title('λ vs TRI (Color: Topography)')
    ax3.set_xlabel('Terrain Ruggedness Index (TRI)')
    ax3.set_ylabel('λ')
    fig.colorbar(sc, ax=ax3, label='Topography')

    ax4 = fig.add_subplot(gs[1, 1])
    train_ci = 3 / train_lambda
    test_ci = 3 / test_lambda
    sns.histplot(train_ci, bins=50, kde=True, color='blue', alpha=0.5, label='Train', ax=ax4, stat='density')
    sns.histplot(test_ci, bins=50, kde=True, color='red', alpha=0.5, label='Test', ax=ax4, stat='density')
    ax4.set_title('95% Confidence Interval Size Distribution')
    ax4.set_xlabel('CI Size (3/λ)')
    ax4.set_ylabel('Density')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'plot_{order}/lambda_analysis_{order}.png', dpi=300)
    plt.show()


plot_lambda_analysis()


# ------------------------------------------------ 5. 空间不确定性可视化
def plot_spatial_uncertainty():
    """在空间网格上可视化不确定性"""
    lambda_grid = np.zeros(target_size)
    lambda_grid[mask_train] = labda_tensor_train.detach().numpy()
    lambda_grid[mask_test] = labda_tensor_test.detach().numpy()

    ci_grid = 3 / lambda_grid
    ci_grid[lambda_grid == 0] = 0

    residual_grid = np.zeros(target_size)
    residual_grid[mask_train] = (phi_tensor_train - ship_tensor_train).abs().detach().numpy()
    residual_grid[mask_test] = (phi_tensor_test - ship_tensor_test).abs().detach().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axs[0].imshow(lambda_grid, cmap='viridis', origin='upper')
    axs[0].set_title('Spatial Distribution of λ')
    fig.colorbar(im0, ax=axs[0], label='λ Value')

    im1 = axs[1].imshow(ci_grid, cmap='plasma', origin='upper')
    axs[1].set_title('Spatial Distribution of 95% CI Size (3/λ)')
    fig.colorbar(im1, ax=axs[1], label='CI Size')

    im2 = axs[2].imshow(residual_grid, cmap='hot', origin='upper')
    axs[2].set_title('Spatial Distribution of |Residual|')
    fig.colorbar(im2, ax=axs[2], label='|φ - ship|')

    plt.tight_layout()
    plt.savefig(f'plot_{order}/spatial_uncertainty_{order}.png', dpi=300)
    plt.show()


plot_spatial_uncertainty()


# ------------------------------------------------ 6. 不确定性带可视化
def plot_uncertainty_bands():
    """展示预测水深及其不确定性区间"""
    sorted_idx = np.argsort(phi_tensor_test.detach().numpy())
    sorted_phi = phi_tensor_test.detach().numpy()[sorted_idx]
    sorted_ship = ship_tensor_test.detach().numpy()[sorted_idx]
    sorted_lambda = labda_tensor_test.detach().numpy()[sorted_idx]

    ci_lower = sorted_phi - 3 / sorted_lambda
    ci_upper = sorted_phi + 3 / sorted_lambda

    plt.figure(figsize=(12, 8))
    plt.fill_between(range(len(sorted_phi)), ci_lower, ci_upper,
                     color='skyblue', alpha=0.4, label='95% Confidence Interval')
    plt.plot(sorted_phi, 'b-', linewidth=1, label='Predicted Depth (φ)')
    plt.plot(sorted_ship, 'ro', markersize=2, alpha=0.5, label='Actual Depth (ship)')
    outside_ci = (sorted_ship < ci_lower) | (sorted_ship > ci_upper)
    plt.plot(np.where(outside_ci)[0], sorted_ship[outside_ci], 'rx', markersize=4, label='Outside CI')
    plt.title('Predicted Depth with Uncertainty Bands (Test Set)')
    plt.xlabel('Sample Index (Sorted by φ)')
    plt.ylabel('Depth')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plot_{order}/uncertainty_bands_{order}.png', dpi=300)
    plt.show()


plot_uncertainty_bands()


# ------------------------------------------------ 7. 综合性能报告
def generate_performance_report():
    """生成综合性能报告"""
    train_phi = phi_tensor_train.detach().numpy()
    train_ship = ship_tensor_train.detach().numpy()
    train_lambda = labda_tensor_train.detach().numpy()
    train_res = np.abs(train_phi - train_ship)

    test_phi = phi_tensor_test.detach().numpy()
    test_ship = ship_tensor_test.detach().numpy()
    test_lambda = labda_tensor_test.detach().numpy()
    test_res = np.abs(test_phi - test_ship)

    metrics = {
        'Train R²': r2_score(train_ship, train_phi),
        'Test R²': r2_score(test_ship, test_phi),
        'Train MAE': mean_absolute_error(train_ship, train_phi),
        'Test MAE': mean_absolute_error(test_ship, test_phi),
        'Train RMSE': np.sqrt(mean_squared_error(train_ship, train_phi)),
        'Test RMSE': np.sqrt(mean_squared_error(test_ship, test_phi)),
        'Train λ Mean': np.mean(train_lambda),
        'Test λ Mean': np.mean(test_lambda),
        'Train Coverage (%)': np.mean(train_res <= (3 / train_lambda)) * 100,
        'Test Coverage (%)': np.mean(test_res <= (3 / test_lambda)) * 100,
        'Correlation (λ, |residual|) Train': np.corrcoef(train_lambda, train_res)[0, 1],
        'Correlation (λ, |residual|) Test': np.corrcoef(test_lambda, test_res)[0, 1]
    }

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE REPORT")
    print("=" * 50)
    for metric, value in metrics.items():
        if 'R²' in metric or 'Correlation' in metric:
            print(f"{metric}: {value:.4f}")
        elif 'Coverage' in metric:
            print(f"{metric}: {value:.2f}%")
        else:
            print(f"{metric}: {value:.2f}")

    plt.figure(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    plt.barh(metric_names, metric_values, color=colors)
    for i, v in enumerate(metric_values):
        if 'Coverage' in metric_names[i]:
            plt.text(v + 0.01, i, f"{v:.2f}%", va='center')
        else:
            plt.text(v + 0.01, i, f"{v:.4f}", va='center')
    plt.title('Model Performance Metrics')
    plt.xlabel('Value')
    plt.xlim(0, max(metric_values) * 1.2)
    plt.tight_layout()
    plt.savefig(f'plot_{order}/performance_report_{order}.png', dpi=300)
    plt.show()

    return metrics


performance_metrics = generate_performance_report()