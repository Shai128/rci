import argparse
import ast
import json

hyperparameters_json = json.loads(open("hyperparameters.json", 'r').read())
HYPERPARAMETERS_TO_USE = hyperparameters_json['hyperparameters']
GAMMAS_TO_USE = hyperparameters_json['gammas']

CALIBRATION_METHODS = ['ACI+CQR', 'RCI+CQR with cal', 'RCI_Y', "RCI_Stretched_Y",
                       "RCI_Stretched_Exp_e_Y", "RCI_Stretched_Exp_5_Y"]

class Hyperparameters:

    @property
    def calibration_method(self):
        return self._calibration_method

    @calibration_method.setter
    def calibration_method(self, calibration_method):
        self._calibration_method = calibration_method
        self.cal_split = ('ACI' in calibration_method and self.is_calibrated) or self.cal_split

    def __init__(self, is_calibrated=True, calibration_method='', lstm_hd=128, lstm_nl=1,
                 lstm_in_hd=[32, 64], lstm_out_hd=[32], do=0.1,
                 lr=0.0005, bs=1, train_all_q=False, gamma=0.05,
                 cal_split=False, z_dim=3, method_type='baseline'):
        self.is_calibrated = is_calibrated
        self.lstm_hd = lstm_hd
        self.lstm_nl = lstm_nl
        self.lstm_in_hd = lstm_in_hd
        self.lstm_out_hd = lstm_out_hd
        self.do = do
        self.lr = lr
        self.bs = bs
        self.train_all_q = train_all_q
        self.gamma = gamma
        self.cal_split = bool(cal_split)
        if is_calibrated:
            self.cal_split = ('ACI' in calibration_method and self.is_calibrated) or (
                        self.cal_split and 'ACI' not in calibration_method)
        self.calibration_method = calibration_method

        self.z_dim = z_dim
        self.method_type = method_type

    def __getitem__(self, item):
        return vars(self)[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def from_args(args, is_calibrated):
        gamma = args.gamma if is_calibrated else 0
        calibration_method = args.calibration_method if is_calibrated else ''
        return Hyperparameters(
                                calibration_method=calibration_method,
                                lstm_hd=args.lstm_hidden_size,
                               lstm_nl=args.lstm_layers,
                               lstm_in_hd=args.lstm_in_layers, lstm_out_hd=args.lstm_out_layers, do=args.dropout,
                               lr=args.lr, bs=args.bs, train_all_q=args.train_all_q,
                               cal_split=args.cal_split, gamma=gamma,
                                is_calibrated=is_calibrated, method_type=args.method_type, z_dim=args.z_dim)
    @staticmethod
    def get_cal_method(calibration_method, cal_split):
        if 'ACI' not in calibration_method:
            return calibration_method
        if not cal_split and 'ACI' in calibration_method:
            return calibration_method.replace("ACI", 'RCI') + ' with cal'
        return calibration_method

    def to_folder_name(self):
        calibration_method = self.calibration_method
        cal_split = self.cal_split
        if self.is_calibrated:
            cal_split = True if 'ACI' in calibration_method else False
            calibration_method = Hyperparameters.get_cal_method(calibration_method, cal_split)
            calibration_prefix = f'calibrated_{calibration_method}_'
        else:
            calibration_prefix = ''
        cal_split_suffix = '_cal_split' if cal_split else ''
        calibration_suffix = f'_gamma={self.gamma}' if self.is_calibrated else ''

        train_all_q = self.train_all_q

        if self.method_type == 'baseline':
            method_params = f"baseline_all_q={int(train_all_q)}_lstm_hd={self['lstm_hd']}_lstm_nl={self['lstm_nl']}_" + \
               f"lstm_in_hd={self['lstm_in_hd']}_lstm_out_hd={self['lstm_out_hd']}_"
        elif self.method_type == 'mqr':
            method_params = f"mqr_z_dim={self.z_dim}_"
        else:
            raise Exception("invalid method type")

        return f"{calibration_prefix}{method_params}" + \
               f"lr={self['lr']}{calibration_suffix}{cal_split_suffix}"


def get_best_hyperparams(hyperparams: Hyperparameters, dataset, is_real):
    hyperparameters_json = json.loads(open("hyperparameters.json", 'r').read())
    HYPERPARAMETERS_TO_USE = hyperparameters_json['hyperparameters']

    cal_split = hyperparams.cal_split
    hp = HYPERPARAMETERS_TO_USE[hyperparams.method_type][f"train_all_q={int(hyperparams.train_all_q)}"][
        f"cal_split={int(cal_split)}"][dataset]
    if type(hp['lstm_in_layers']) == str:
        hp['lstm_in_layers'] = ast.literal_eval(hp['lstm_in_layers'])
    if type(hp['lstm_out_layers']) == str:
        hp['lstm_out_layers'] = ast.literal_eval(hp['lstm_out_layers'])

    return hp


def get_best_gamma(hyperparams: Hyperparameters, dataset, is_real):
    hyperparameters_json = json.loads(open("hyperparameters.json", 'r').read())
    GAMMAS_TO_USE = hyperparameters_json['gammas']
    try:
        return float(GAMMAS_TO_USE[dataset][f'train_all_q={int(hyperparams.train_all_q)}'][hyperparams.calibration_method])
    except Exception:
        print(f"didn't find best gamma for data {dataset} and cal method: {hyperparams.calibration_method}")
        raise

def get_best_hyperparams_from_args(args):
    try:
        hyperparams = Hyperparameters.from_args(args, False)
        hyperparameters_to_load = get_best_hyperparams(hyperparams, args.dataset_name, args.ds_type.lower() == 'real')
        args = argparse.Namespace(**{**vars(args), **hyperparameters_to_load})
    except Exception as e:
        print(f"didn't find best hyperparameters for dataset: {args.dataset_name} because {e}")
        pass
    return args
