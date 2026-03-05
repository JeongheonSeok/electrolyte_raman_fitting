import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
from lmfit import Model, models
from tqdm import tqdm
import os
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

DATA_PATH = "./spectra"
ABSTRACT_RESULT_PATH = "./results/abstract_results.csv"
FITTED_FUNCTION_PATH = "./results/fitted_functions.csv"
CURVE_IMAGE_PATH = "./results/curve_images"
PEAK_LOC_PATH = "./peak_location.csv"
SETTINGS_PATH = "./settings_better-ing.csv"

def get_peak_dict(peak_loc_path=PEAK_LOC_PATH):
    with open(peak_loc_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        peak_dict = {}
        for row in reader:
            peak_dict[row[0]] = [float(row[1]), float(row[2])]
        if len(peak_dict) == 0:
            raise ValueError("No peak information found in the file")
    return peak_dict

def get_settings_dict(settings_path=SETTINGS_PATH):
    with open(settings_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        settings_dict = {}
        for row in reader:
            settings_dict[row[0]] = float(row[1])
    return settings_dict

def construct_model(peak_dict, settings_dict):
    peak_names = list(peak_dict.keys())
    peak_type = settings_dict['PEAK_TYPE']
    if peak_type == 0:
        from lmfit.models import GaussianModel as PeakModel
    elif peak_type == 1:
        from lmfit.models import LorentzianModel as PeakModel
    elif peak_type == 2:
        from lmfit.models import VoigtModel as PeakModel
    else:
        raise ValueError("Invalid peak type")
    model = PeakModel(prefix=peak_names[0]+'_')
    for peak_name in peak_names[1:]:
        model += PeakModel(prefix=peak_name+'_')

    background_type = settings_dict['BACKGROUND']
    if background_type == 0:
        pass
    elif background_type == 1:
        model += models.ConstantModel(prefix='constbg_')
    elif background_type == 2:
        model += models.LinearModel(prefix='linbg_')
    elif background_type == 3:
        model += models.LinearModel(prefix='linbg1_') + models.LinearModel(prefix='linbg2_')
    else:
        raise ValueError("Invalid background type")
    
    params = model.make_params()
    for peak_name in peak_names:
        params[peak_name+'_amplitude'].set(min=0)
        params[peak_name+'_center'].set((peak_dict[peak_name][0] + peak_dict[peak_name][1])/2, min=peak_dict[peak_name][0], max=peak_dict[peak_name][1])
        params[peak_name+'_sigma'].set(
            (settings_dict['SIGMA_LB'] + settings_dict['SIGMA_UB']) / 2,
            min=settings_dict['SIGMA_LB'],
            max=settings_dict['SIGMA_UB']
        )
    
    return model, params


def initialize_params_from_data(params, peak_dict, x, y):
    # Initialize amplitude from actual y values near each peak range
    y_baseline = float(np.percentile(y, 10))
    for peak_name, (lo, hi) in peak_dict.items():
        mask = (x >= lo) & (x <= hi)
        if mask.any():
            amp_init = float(np.max(y[mask]) - y_baseline)
        else:
            amp_init = float(np.max(y) - y_baseline)
        amp_init = max(amp_init, 1e-6)
        params[peak_name + '_amplitude'].set(amp_init)
    return params


def _run_single_trial(args):
    # Worker for ThreadPoolExecutor; each call runs one local fit
    model, trial_params, x, y = args
    return model.fit(y, trial_params, x=x, method='leastsq')


def fit_robust(model, params, x, y, settings_dict):
    # Stage 1: global search via Differential Evolution
    # Stage 2: refine with Levenberg-Marquardt starting from DE result
    # Optional: parallel multi-start (add USE_MULTISTART and N_STARTS to settings.csv)
    use_multistart = int(settings_dict.get('USE_MULTISTART', 0)) == 1
    n_starts = int(settings_dict.get('N_STARTS', 20))

    best_result = None
    best_rsq = -np.inf

    # Strategy A: Differential Evolution for global search
    try:
        res_de = model.fit(y, params, x=x, method='differential_evolution')
        res_refined = model.fit(y, res_de.params, x=x, method='leastsq')
        if res_refined.rsquared > best_rsq:
            best_rsq = res_refined.rsquared
            best_result = res_refined
    except Exception:
        pass

    # Strategy B: parallel multi-start, randomizing only center and sigma (optional)
    # Amplitude is kept from data-based initialization since it is reliably estimated,
    # while center and sigma are the uncertain parameters when peaks overlap.
    # scipy/numpy release the GIL during computation, so ThreadPoolExecutor gives
    # true parallelism here without the pickling overhead of ProcessPoolExecutor.
    if use_multistart:
        rng = np.random.default_rng(42)
        trial_args = []
        for _ in range(n_starts):
            tp = params.copy()
            for name, par in tp.items():
                if par.vary and par.min is not None and par.max is not None:
                    if not name.endswith('_amplitude'):
                        tp[name].set(float(rng.uniform(par.min, par.max)))
            trial_args.append((model, tp, x, y))

        n_workers = min(n_starts, multiprocessing.cpu_count())
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_single_trial, args) for args in trial_args]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res.rsquared > best_rsq:
                        best_rsq = res.rsquared
                        best_result = res
                except Exception:
                    continue

    # Fallback: plain local optimization
    if best_result is None:
        best_result = model.fit(y, params, x=x)

    return best_result


def save_curve_image(x, y, result, filename, peak_dict, settings_dict, curve_image_path=CURVE_IMAGE_PATH, FNC_DOTS=200):
    if not os.path.exists(curve_image_path):
        os.makedirs(curve_image_path, exist_ok=True)
    
    fig, (ax_res, ax_main) = plt.subplots(
        2, 1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 3]}
    )
    # Plot residuals
    residuals = result.residual
    ax_res.scatter(x, residuals)
    ax_res.axhline(0, linestyle='--')
    ax_res.set_ylabel('residuals')

    # Plot main figure
    ax_main.scatter(x, y, label='measured data')
    
    x_fit = np.linspace(settings_dict['RANGE_MIN'], settings_dict['RANGE_MAX'], FNC_DOTS)
    y_fit = result.model.eval(params=result.params, x=x_fit)
    fit_label = f"fit (R²={result.rsquared:.4f})"
    ax_main.plot(x_fit, y_fit, '-', label=fit_label, color='#a0a0a0')

    peak_names = list(peak_dict.keys())
    peak_amplitudes = [result.best_values[peak_name + '_amplitude'] for peak_name in peak_names]
    total_amplitude = sum(peak_amplitudes)
    peak_ratios = {}
    if total_amplitude != 0:
        for name, amp in zip(peak_names, peak_amplitudes):
            peak_ratios[name] = amp / total_amplitude * 100

    # component curves (peak + background)
    y_fit_components = result.model.eval_components(params=result.params, x=x_fit)
    for i in range(len(y_fit_components)):
        prefixname = result.model.components[i].prefix
        comp_name = prefixname[:-1]
        if comp_name in peak_ratios:
            label = f"{comp_name} ({peak_ratios[comp_name]:.1f}%)"
            center = result.best_values[comp_name + '_center']
            ax_main.axvline(center, color='gray', linestyle='--', alpha=0.6)
        else:
            label = comp_name
        ax_main.plot(x_fit, y_fit_components[prefixname], '-', label=label)

    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    mae = np.mean(np.abs(residuals))
    ax_res.set_title(f'Residuals (MAE={mae:.4g})')
    ax_main.set_title('Spectrum and fit')
    fig.suptitle(filename, fontsize=18, fontweight='bold', y=0.975)

    ax_main.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(os.path.join(curve_image_path, filename + '.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)  # release memory; critical when running many workers in parallel


def process_file(args):
    # Top-level function required for ProcessPoolExecutor (must be picklable)
    filename, data_path, peak_dict, settings_dict, curve_image_path = args

    file_path = os.path.join(data_path, filename)
    spectrum = np.loadtxt(file_path, delimiter=',')
    x_all = spectrum[:, 0]; y_all = spectrum[:, 1]
    mask = (x_all >= settings_dict['RANGE_MIN']) & (x_all <= settings_dict['RANGE_MAX'])
    x = x_all[mask]; y = y_all[mask]

    model, params = construct_model(peak_dict, settings_dict)
    params = initialize_params_from_data(params, peak_dict, x, y)
    result = fit_robust(model, params, x, y, settings_dict)

    save_curve_image(x, y, result, filename, peak_dict, settings_dict, curve_image_path)

    # Return plain data so the main process can write CSVs without race conditions
    peak_names = list(peak_dict.keys())
    best_values = dict(result.best_values)
    best_values['rsquared'] = result.rsquared
    return filename, peak_names, best_values


def write_csv_results(results, peak_dict):
    # Write abstract results (amplitude ratios)
    os.makedirs(os.path.dirname(ABSTRACT_RESULT_PATH), exist_ok=True)
    peak_names = list(peak_dict.keys())
    with open(ABSTRACT_RESULT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename'] + [p + '(%)' for p in peak_names])
        for filename, _, best_values in results:
            amps = [best_values[p + '_amplitude'] for p in peak_names]
            total = sum(amps)
            writer.writerow([filename] + [a / total * 100 for a in amps])

    # Write full fitted parameters
    param_keys = [k for k in results[0][2].keys() if k != 'rsquared']
    with open(FITTED_FUNCTION_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename'] + param_keys + ['rsquared'])
        for filename, _, best_values in results:
            writer.writerow([filename] + [best_values[k] for k in param_keys] + [best_values['rsquared']])


if __name__ == "__main__":
    file_names = [f for f in os.listdir(DATA_PATH)
                  if os.path.isfile(os.path.join(DATA_PATH, f))]

    peak_dict = get_peak_dict()
    settings_dict = get_settings_dict()
    os.makedirs(CURVE_IMAGE_PATH, exist_ok=True)

    args_list = [
        (fn, DATA_PATH, peak_dict, settings_dict, CURVE_IMAGE_PATH)
        for fn in file_names
    ]

    # Each file is processed independently, so parallelize across all CPU cores
    n_workers = multiprocessing.cpu_count()
    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_file, args): args[0] for args in args_list}
        with tqdm(total=len(file_names), desc="Processing files") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append((filename, e))
                pbar.update(1)
                pbar.set_postfix_str(filename)

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for fn, e in errors:
            print(f"  {fn}: {e}")

    if results:
        write_csv_results(results, peak_dict)
        print(f"\nDone. {len(results)} file(s) processed successfully.")
