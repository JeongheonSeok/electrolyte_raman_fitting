import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, models
#import pandas as pd
from tqdm import tqdm
import os
import csv

DATA_PATH = "./spectra"
ABSTRACT_RESULT_PATH = "./results/abstract_results.csv"
FITTED_FUNCTION_PATH = "./results/fitted_functions.csv"
CURVE_IMAGE_PATH = "./results/curve_images"
PEAK_LOC_PATH = "./peak_location.csv"
SETTINGS_PATH = "./settings.csv"

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
        params[peak_name+'_center'].set(min=peak_dict[peak_name][0], max=peak_dict[peak_name][1])
    
    return model, params

def save_abstract_result(result, filename, peak_dict, abstract_result_path=ABSTRACT_RESULT_PATH):
    # make ABSTRACT_RESULT_PATH directory if not exists
    if not os.path.exists(os.path.dirname(abstract_result_path)):
        os.makedirs(os.path.dirname(abstract_result_path))
    
    peak_names = list(peak_dict.keys())

    # Write header if not exists
    header_exists = False
    if os.path.exists(abstract_result_path):
        with open(abstract_result_path, 'r', newline='') as f:
            first_line = f.readline()
            if first_line:
                header_exists = True
    if not header_exists:
        header = ['filename']
        for peak_name in peak_names:
            header.append(peak_name+'(%)')
        with open(abstract_result_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    # Write peak amplitude ratio to the file
    peak_amplitudes = [result.best_values[peak_name + '_amplitude'] for peak_name in peak_names]
    total_amplitude = sum(peak_amplitudes)
    ratio = [peak_amplitudes[i] / total_amplitude * 100 for i in range(len(peak_amplitudes))]
    with open(abstract_result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename] + ratio)

def save_fitted_function(result, filename, peak_dict, fitted_function_path=FITTED_FUNCTION_PATH):
    # Write header if not exists
    header_exists = False
    if os.path.exists(fitted_function_path):
        with open(fitted_function_path, 'r', newline='') as f:
            first_line = f.readline()
            if first_line:
                header_exists = True
    if not header_exists:
        bvk = list(result.best_values.keys())
        bvk.append('rsquared')
        header = ['filename'] + bvk
        with open(fitted_function_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    # Write fitted function to the file
    with open(fitted_function_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        header_keys = header[1:]
    fitted_function = [result.best_values[bk] for bk in header_keys[:-1]]
    with open(fitted_function_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([filename] + fitted_function + [result.rsquared])

def save_curve_image(x, y, result, filename, peak_dict, settings_dict, curve_image_path=CURVE_IMAGE_PATH, FNC_DOTS=200):
    # make CURVE_IMAGE_PATH directory if not exists
    if not os.path.exists(curve_image_path):
        os.makedirs(curve_image_path, exist_ok=True)
    
    fig, (ax_res, ax_main) = plt.subplots(
        2, 1,
        figsize=(11,19),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 3]}
    )
    # Plot residuals
    residuals = result.residual
    ax_res.scatter(x, residuals)
    ax_res.axhline(0, linestyle='--')
    ax_res.set_ylabel('residuals')

    # Plot main figure
    x_fit = np.linspace(settings_dict['RANGE_MIN'], settings_dict['RANGE_MAX'], FNC_DOTS)
    y_fit = result.model.eval(params=result.params, x=x_fit)
    ax_main.plot(x_fit, y_fit, '-', label='fit', color='gray')
    y_fit_components = result.model.eval_components(params=result.params, x=x_fit)
    for i in range(len(y_fit_components)):
        prefixname = result.model.components[i].prefix
        ax_main.plot(x_fit, y_fit_components[prefixname], '-', label=prefixname[:-1])

    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.legend()
    fig.savefig(os.path.join(curve_image_path, filename[:-4] + '.png'))
    fig.close()

if __name__ == "__main__":
    file_names = os.listdir(DATA_PATH)

    peak_dict = get_peak_dict()
    settings_dict = get_settings_dict()

    with tqdm(file_names, desc="Processing files") as pbar:
        for filename in pbar:
            file_path = os.path.join(DATA_PATH, filename)
            if not os.path.isfile(file_path):
                continue
            pbar.set_postfix_str(filename)

            spectrum = np.loadtxt(file_path, delimiter=',')
            x_all = spectrum[:, 0]; y_all = spectrum[:, 1]
            mask =  (x_all >= settings_dict['RANGE_MIN']) & (x_all <= settings_dict['RANGE_MAX'])
            x = x_all[mask]; y = y_all[mask]

            model, params = construct_model(peak_dict, settings_dict)

            result = model.fit(y, params, x=x)

            save_abstract_result(result, filename, peak_dict)
            save_fitted_function(result, filename, peak_dict)
            save_curve_image(x, y, result, filename, peak_dict, settings_dict)