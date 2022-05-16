import matplotlib
import torch
import matplotlib.pyplot as plt
from scipy.linalg import circulant

from helper import create_folder_if_it_doesnt_exist
from utils.Calibration.ACICalibration import ACICalibration


def display_plot(x_label=None, y_label=None, title=None, display_legend=False, save_path=None, dpi=300):
    if display_legend:
        plt.legend()

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    plt.show()


def plot_syn_data_intervals(y_test, y_samples, upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                            initial_time, T, initial_plotting_time, save_dir):
    times = list(range(initial_time + initial_plotting_time, initial_time + initial_plotting_time + T))

    if y_samples is not None:
        plt.scatter(times, y_samples[0][initial_plotting_time:initial_plotting_time + T].cpu(), color='blue')

        times_rep = torch.Tensor(times).unsqueeze(0).repeat(100, 1).flatten()
        plt.scatter(times_rep, y_samples[:100, initial_plotting_time:initial_plotting_time + T].cpu(), color='blue')
        #
        # for i in range(1, 100):
        #     plt.scatter(times, y_samples[i][initial_plotting_time:initial_plotting_time + T].cpu(), color='blue')

    plt.plot(times, y_test[initial_plotting_time:initial_plotting_time + T].cpu(), color='green')

    if true_upper_q is not None and true_lower_q is not None:
        plt.plot(times, true_upper_q[initial_plotting_time:initial_plotting_time + T].cpu(), color='red', linewidth=4,
                 label='True quantiles')
        plt.plot(times, true_lower_q[initial_plotting_time:initial_plotting_time + T].cpu(), color='red', linewidth=4)

    plt.plot(times, lower_q_pred[initial_plotting_time:initial_plotting_time + T].cpu(), color='purple',
             label='Estimated quantiles', linewidth=4)
    plt.plot(times, upper_q_pred[initial_plotting_time:initial_plotting_time + T].cpu(), color='purple', linewidth=4)

    save_file_name = f"intervals_initial_plotting_time={initial_time + initial_plotting_time}_T={T}.png"
    display_plot(x_label="Time", y_label="Y", display_legend=True,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_syn_data_interval_length(upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                                  initial_time, T, initial_plotting_time, save_dir):
    times = list(range(initial_time + initial_plotting_time, initial_time + initial_plotting_time + T))

    if true_upper_q is not None and true_lower_q is not None:
        plt.plot(times, (true_upper_q[:T] - true_lower_q[:T]).cpu(), color='red', linewidth=4, label='True quantiles',
                 alpha=0.5)
    plt.plot(times, (upper_q_pred[:T] - lower_q_pred[:T]).cpu(), color='purple', linewidth=4,
             label='Estimated quantiles',
             alpha=0.5)
    save_file_name = f"intervals_length_initial_plotting_time={initial_time + initial_plotting_time}_T={T}.png"
    matplotlib.rc('font', **{'size': 16})
    display_plot(x_label="Time", y_label="Interval's length", display_legend=True,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_syn_data_results(y_test, y_samples, upper_q_pred, lower_q_pred, true_upper_q, true_lower_q, initial_time,
                          save_dir, args, calibration: ACICalibration = None):
    if args.suppress_plots:
        return
    create_folder_if_it_doesnt_exist(save_dir)

    for T in [min(1000, len(y_test)), min(50, len(y_test)), len(y_test)]:
        for initial_plotting_time in [0, 400, len(y_test) - 1050]:
            if len(y_test) < initial_plotting_time + T or initial_plotting_time < 0:
                continue
            if calibration is not None:
                calibration.plot_parameter_vs_time(initial_time+initial_plotting_time,
                                                   initial_time+initial_plotting_time+T, save_dir=save_dir)
            plot_syn_data_intervals(y_test, y_samples, upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                                    initial_time, T, initial_plotting_time, save_dir)
            plot_syn_data_interval_length(upper_q_pred, lower_q_pred, true_upper_q, true_lower_q,
                                          initial_time, T, initial_plotting_time, save_dir)
            
    lengths = (upper_q_pred - lower_q_pred).cpu().numpy()
    plt.hist(lengths, bins=lengths.shape[0] // 100)
    save_file_name = f"interval_length_histogram.png"
    display_plot(x_label="Interval's length", y_label="Count", display_legend=False,
                 save_path=f'{save_dir}/{save_file_name}')

def plot_local_coverage(coverage, local_coverage, initial_time, save_dir):
    matplotlib.rc('font', **{'size': 21})
    plt.figure(figsize=(7,4.5))
    times = list(range(initial_time, initial_time + local_coverage.shape[0]))
    plt.plot(times, local_coverage)
    plt.axhline(coverage.float().mean().item(), ls='--')
    save_file_name = f"local coverage level.png"
    display_plot(x_label="Time", y_label="Local Coverage Level", display_legend=False,
                 save_path=f'{save_dir}/{save_file_name}')


def plot_mqr_results(y_test, quantile_regions, initial_time, alpha, figures_save_dir, args, calibration):
    if args.suppress_plots:
        return
    create_folder_if_it_doesnt_exist(figures_save_dir)

    coverage = quantile_regions.is_in_region(y_test, is_scaled=False)
    local_size = 500
    coverage_mat = circulant(coverage.float().cpu().numpy())
    local_coverage = coverage_mat[:local_size // 2, local_size // 2:-local_size // 2].mean(axis=0)
    plot_local_coverage(coverage, local_coverage, initial_time + local_size // 2, figures_save_dir)

