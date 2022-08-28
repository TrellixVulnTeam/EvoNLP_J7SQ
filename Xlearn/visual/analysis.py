from pathlib import Path
from inspect import cleandoc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna, re
import plotly.io as pio
from operator import itemgetter
from optuna.study._study_direction import StudyDirection
from Xlearn.utils.io import pickle_load, yaml_load, hashcode

__all__ = ['write_analysis', 'OptunaAnalysis']

def smooth_filter(x, window_length, polyorder, **kwargs):
    r"""Smooth a sequence of noisy data points by applying `Savitzky–Golay filter`_. It uses least
    squares to fit a polynomial with a small sliding window and use this polynomial to estimate
    the point in the center of the sliding window. 
    
    This is useful when a curve is highly noisy, smoothing it out leads to better visualization quality.
    
    .. _Savitzky–Golay filter:
        https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    
    Example:
    
        >>> import matplotlib.pyplot as plt
    
        >>> x = np.linspace(0, 4*2*np.pi, num=100)
        >>> y = x*(np.sin(x) + np.random.random(100)*4)
        >>> y2 = smooth_filter(y, window_length=31, polyorder=10)
        
        >>> plt.plot(x, y)
        >>> plt.plot(x, y2, 'red')
        
    Args:
        x (list): one-dimensional vector of scalar data points of a curve. 
        window_length (int): the length of the filter window
        polyorder (int): the order of the polynomial used to fit the samples
        
    Returns:
        ndarray: an numpy array of smoothed curve data
    """
    from scipy.signal import savgol_filter
    x = np.asarray(x)
    assert x.ndim == 1, 'only a single vector of scalar values is supported'
    out = savgol_filter(x, window_length, polyorder, **kwargs)
    return out


def interp_curves(x, y):
    r"""Piecewise linear interpolation of a discrete set of data points and generate new :math:`x-y` values
    from the interpolated line. 
    
    It receives a batch of curves with :math:`x-y` values, a global min and max of the x-axis are 
    calculated over the entire batch and new x-axis values are generated to be applied to the interpolation
    function. Each interpolated curve will share the same values in x-axis. 
    
    .. note::
    
        This is useful for plotting a set of curves with uncertainty bands where each curve
        has data points at different :math:`x` values. To generate such plot, we need the set of :math:`y` 
        values with consistent :math:`x` values. 
        
    .. warning::
    
        Piecewise linear interpolation often can lead to more realistic uncertainty bands. Do not
        use polynomial interpolation which the resulting curve can be extremely misleading. 
    
    Example::
    
        >>> import matplotlib.pyplot as plt
    
        >>> x1 = [4, 5, 7, 13, 20]
        >>> y1 = [0.25, 0.22, 0.53, 0.37, 0.55]
        >>> x2 = [2, 4, 6, 7, 9, 11, 15]
        >>> y2 = [0.03, 0.12, 0.4, 0.2, 0.18, 0.32, 0.39]
        
        >>> plt.scatter(x1, y1, c='blue')
        >>> plt.scatter(x2, y2, c='red')
        
        >>> new_x, new_y = interp_curves([x1, x2], [y1, y2], num_point=100)
        >>> plt.plot(new_x[0], new_y[0], 'blue')
        >>> plt.plot(new_x[1], new_y[1], 'red')
        
    Args:
            x (list): a batch of x values. 
            y (list): a batch of y values. 
            num_point (int): number of points to generate from the interpolated line. 
    
    Returns:
        tuple: a tuple of two lists. A list of interpolated x values (shared for the batch of curves)
            and followed by a list of interpolated y values. 
    """
    new_x = np.unique(np.hstack(x))
    assert new_x.ndim == 1
    ys = [np.interp(new_x, curve_x, curve_y) for curve_x, curve_y in zip(x, y)]
    return new_x, ys

def set_ticker(ax, axis='x', num=None, KM_format=False, integer=False):
    if axis == 'x':
        axis = ax.xaxis
    elif axis == 'y':
        axis = ax.yaxis
    if num is not None:
        axis.set_major_locator(plt.MaxNLocator(num))
    if KM_format:
        def tick_formatter(x, pos):
            if abs(x) >= 0 and abs(x) < 1000:
                return int(x) if integer else x
            elif abs(x) >= 1000 and abs(x) < 1000000:
                return f'{int(x/1000)}K' if integer else f'{x/1000}K'
            elif abs(x) >= 1000000:
                return f'{int(x/1000000)}M' if integer else f'{x/1000000}M'
        axis.set_major_formatter(plt.FuncFormatter(tick_formatter))
    return ax


def read_xy(log_folder, file_name, get_x, get_y, smooth_out=False, smooth_kws=None, point_step=1):
    glob_dir = lambda x: [p for p in x.glob('*/') if p.is_dir() and str(p.name).isdigit()]
    dfs = []
    for id_folder in glob_dir(Path(log_folder)):
        x = []
        y = []
        try:
            for seed_folder in glob_dir(id_folder):
                logs = pickle_load(seed_folder / file_name)
                x.append([get_x(log) for log in logs])
                y.append([get_y(log) for log in logs])
        except FileNotFoundError:
            continue
            
        new_x, ys = interp_curves(x, y)  # all seeds share same x values
        
        if smooth_out:
            if smooth_kws is None:
                smooth_kws = {'window_length': 20, 'polyorder': 3}
            ys = [smooth_filter(y, **smooth_kws) for y in ys]
        
        if point_step > 1:
            idx = np.arange(0, new_x.size, step=point_step)
            new_x = new_x[idx, ...]
            ys = [y[idx, ...] for y in ys]

        df = pd.DataFrame({'x': np.tile(new_x, len(ys)), 'y': np.hstack(ys)})
        config = yaml_load(id_folder / 'config.yml')
        config = pd.DataFrame([config.values()], columns=config.keys())
        config = config.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
        df = pd.concat([df, config], axis=1, ignore_index=False)
        df = df.fillna(method='pad')  # padding all NaN configs
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0, ignore_index=True)    
    return dfs

def get_configs(log_folder):
    data = pickle_load(Path(log_folder) / 'configs.pkl')
    configs = pd.DataFrame(data, columns=data[0].keys())
    return configs

def get_train_log(log_folder, xlabel, ylabel):
    get_x = lambda log: log[xlabel][0]
    get_y = lambda log: np.mean(log[ylabel])
    file_name = 'train_logs.pkl'
    data = read_xy(log_folder, file_name, get_x, get_y, smooth_out=True)
    return data

def __max_epoch_print(col: pd.Series):
    if np.isnan(col[1]):    # std is none
        return f'{col[0]:.5f}'
    else:
        return f'{col[0]:.5f} ± {col[1]:.5f}'
    
def max_epoch(df, x, y, single=True):
    df[y] = df[y].astype('float32')
    if single: 
        y = [y] 
    max_id = df.groupby(['ID','seed'])[y[0]].idxmax()
    max_res = df.loc[max_id, [*y, x]]
    epoch = max_res[x].mean()
    if single:
        max, mean, std, median, min = max_res[y[0]].agg(['max','mean','std','median','min'])
        return pd.Series({'max': max, 'mean': mean, 'std': std, 'epoch': epoch, 'median': median, 'min': min})
    else:
        out = max_res[y].agg(['mean','std']).apply(__max_epoch_print, axis=0)
        return pd.concat([out, pd.Series({'epoch': epoch})])

def print_eval_log(data: pd.DataFrame, category: str = '', column: str='', path: str='', x: str='x', y: str='value'):
    if path: f = open(path, 'a')
    else: f = None
    def show(df, title):
        subdf = df.groupby(category).apply(max_epoch, x, y, isinstance(y,str))
        if title: print('\n{}'.format(title), file=f)
        print('{:^24s}'.format(subdf.to_markdown()), file=f)
    if column:
        for col, df in data.groupby(column):
            show(df, '%s=%s'%(column, col))
    else:
        show(data, '%s'%(category))
    if f: f.close()


def get_eval_log(log_folder):
    glob_dir = lambda x: [p for p in x.glob('*/') if p.is_dir() and str(p.name).isdigit()]
    dfs = []
    for id_folder in glob_dir(Path(log_folder)):
        for seed_folder in glob_dir(id_folder):
            try:
                logs = pickle_load(seed_folder / 'eval_logs.pkl')
            except FileNotFoundError:
                continue
            logs = [{k:v[0] for k,v in log.items()} for log in logs]
            df = pd.DataFrame(logs)
            a,b = df.filter(regex='^train'), df.filter(regex='^eval')
            a.columns = a.columns.str.replace("train_", "")
            b.columns = b.columns.str.replace("eval_", "")
            df= pd.concat([b.assign(ind="test"), a.assign(ind="train")])
            df['epoch'] = df.index + 1
            df['seed'] = seed_folder.name
            config = yaml_load(id_folder / 'config.yml')
            config = pd.DataFrame([config.values()], columns=config.keys())
            config = config.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
            df = pd.concat([df, config], axis=1, ignore_index=False)
            df = df.fillna(method='pad')  # padding all NaN configs
            dfs.append(df)

    parser = pd.io.parsers.base_parser.ParserBase({'usecols': None})
    for df in dfs: df.columns = parser._maybe_dedup_names(df.columns) 
    dfs = pd.concat(dfs, axis=0, ignore_index=True)    
    return dfs


def make_plot(data: pd.DataFrame, category: str = 'nn.type', column: str='', path: str='', x: str='x', y: str='value', **kwargs):
    import seaborn as sns
    sns.set()
    kwargs = {
        'ci': 50, 'err_kws': {'alpha': 0.2}, 'hue': category, 'kind': 'line', **kwargs
    }
    if isinstance(y, list):
        id_vars = [_ for _ in (x, column, category) if _]
        data = data.melt(id_vars=id_vars, value_vars=y)
        kwargs.update({'data': data, 'x': x, 'y': 'value', 'row': 'variable'})
    else:
        kwargs.update({'data': data, 'x': x, 'y': y})
    data[category].fillna(value=0, inplace=True)
    if column:
        ncol = data[column].nunique() #was used to set colwrap, but this conflict with row
        g = sns.relplot(
            col=column, height=4, aspect=1.5, 
            facet_kws=dict(sharex=True, sharey=False),
            **kwargs
        )
    else:
        g = sns.relplot(**kwargs)
        
    [set_ticker(ax, axis='x', num=6, KM_format=True, integer=True) for ax in g.axes.flatten()]
    g.set_xlabels(x)
    if isinstance(y, list):
        for (ylabel, xlabel), ax in g.axes_dict.items():
            ax.set_ylabel(ylabel)
    else:
        g.set_ylabels(y)
    if path: g.savefig(path)
    return g

def write_analysis(path, logpath, name, metric_key):
    py_str = f"""
from Xlearn.visual import OptunaAnalysis
study = OptunaAnalysis('{name}', 'sqlite:///{logpath}/experiment.db')
study.df
study.make_plot(category='seed', column='', y='{metric_key}')
study.make_coordinate()
    """
    with open(path, 'w') as f:
        f.write(cleandoc(py_str))

def plot_distribution(data: list, chunk_value: float, title: str, path: str = ''):
    import seaborn as sns
    def plot(subdata, ax, title):
        sns.distplot(subdata, ax=ax).set(title=title)
        summary = pd.DataFrame({'count': subdata}).describe().round(2)
        ax.table(
            cellText=summary.values, rowLabels=summary.index, colLabels=summary.columns, cellLoc = 'right', rowLoc = 'center', loc='right', bbox=[.65,.45,.25,.4]
        )   # bbox: (x0, y0, width, height)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    plot(data, axs[0], f'{title} | original complete data')
    chunk_data = [x for x in data if x < chunk_value]
    plot(chunk_data, axs[1], f'{title} | chunked @{chunk_value} data')
    # Pad the saved area by 10% in the x-direction and 10% in the y-direction
    if path: fig.savefig(path)


class OptunaAnalysis:
    """
    :param dbpth:       something like sqlite:///logs/test/experiment.db
    :param name:        study name, should be the same when running the experiment
    """
    def __init__(self, name, dbpath):
        
        self.study = optuna.load_study(study_name=name, storage=dbpath)
        self.get_data()
        self.best_trial = self.study.best_trial
        print(f'Best value {self.best_trial.value} (objective: {self.study.direction.name}\nBest hyperparameters: {self.best_trial.params}')

    @staticmethod
    def pad_data(df: pd.DataFrame):
        maxlen = max(len(x) for sub in df.values for x in sub)
        df = df.applymap(lambda x: x+[x[-1]]*(maxlen-len(x)))
        return df

    def get_data(self):
        """extract all recorded data
        e.g., 
            'value', 'dr1', 'seed', 'val_acc',      'val_f1'
        0   0.992899  0.0   1231.0  [0.98,0.99...]  [0.92,0.93...]
        ...
        after some operation, we finally get
        	seed	value	dr1	ind	acc	f1
        0	1231.0	0.992899	0.0	val	0.98325	0.983081
        1	1231.0	0.992899	0.0	val	0.987083	0.986903
        ...
        """
        # df = self.study.trials_dataframe().drop(['state','datetime_start','datetime_complete','duration'], axis=1)
        # df = df.filter(regex=r'^(?!system_attrs)')  # remove column startswith system_
        df = self.study.trials_dataframe(
            attrs=('number','value','params','user_attrs','state')
        )
        df=df.rename(columns = {'user_attrs_loss':'user_attrs_train_cost'})
        if (state := df.state).nunique() > 1:
            print('======= {0} successful trials and {1} fail trials ======='.format(*state.value_counts().values))
            df = df.query('state=="COMPLETE"').drop('state', axis=1)
        pattern = re.compile('user_attrs_(.*)$')    # match user_attrs_pre-0 ..
        col2explode = [m.group(1) for m in map(pattern.match, df.columns) if m]
        df.columns = [re.sub('params_|sys_attrs_|user_attrs_','',name) for name in df.columns]
        df = df.set_index(list(set(df.columns) - set(col2explode)))
        df = df.explode(col2explode)
        self.params = list(set(df.index.names) - {'value','number'})
        df = df.set_index(df.groupby(df.index.names).cumcount()+1, append=True)
        df.columns = df.columns.str.split('_', expand=True, n=1)
        df = df.apply(pd.to_numeric)
        if df.columns.nlevels > 1:
            df = df.stack(0).rename_axis(df.index.names[:-1]+['epoch','ind'])
        else:
            df = df.rename_axis(df.index.names[:-1]+['epoch'])
        df = df.reset_index().rename(columns={'number': 'ID'})
        self.df = df
        self.param_d = pd.DataFrame.from_records(t.params for t in self.trials)
        for k, l in self.param_d.iteritems():
            setattr(self, k, set(l))

    @property
    def trials(self):
        return self.study.trials

    def make_pivot(self, metric, rows='', columns='', query='', aggfunc='', round=5):
        df = self.df
        if query: 
            df = df.query(query)
        if isinstance(aggfunc, str) and aggfunc.startswith('@'):
            df = df.sort_values(['ID','ind','epoch'], ascending=False)
            obj, tgt = aggfunc.split('@')[1:]
            # assert tgt in ('train','valid','test'), "TODO: current API restrictions"
            if (match := re.match(r'([a-zA-Z]+)(\d+)', obj)):
                obj, topk = match.groups()
                raise NotImplementedError
            if obj == 'max':
                idx = df.groupby(['ID','ind'])[metric].idxmax()
            elif obj == 'min':
                idx = df.groupby(['ID','ind'])[metric].idxmin()
            elif obj == 'last':
                idx = df.reset_index().groupby(['ID','ind'])['index'].first()
            tgt_df = df.loc[idx.filter(regex=tgt, axis=0), ['ID','epoch']]
            df = pd.merge(tgt_df, df, on=['ID', 'epoch'], how='left')
            aggfunc = ''
        if aggfunc == '':
            aggfunc = lambda x: f'{np.mean(x):.{round}f} ± {np.std(x):.{round}f}'
        rows = rows or self.params
        columns = columns or 'ind'
        if isinstance(columns, str) and columns.startswith('@'):
            ind, *columns = columns.split('@')[1:]
            df = df.query(f'ind=="{ind}"')
            df = df.melt(
                id_vars=rows, value_vars=columns, value_name='melt_value'
            )
            columns, metric = 'variable', 'melt_value'
        return df.pivot_table(
            values=metric,index=rows,columns=columns,aggfunc=aggfunc
        ).reset_index()

    def make_plot(self, category: str, column: str = '', y: str = 'loss', path: str = '', x: str = 'epoch', query='', outlier=False, **kwargs):
        df = self.df
        if query:
            df = df.query(query)
        return make_plot(df, category, column, path, x, y, **kwargs)

    def make_best(self, category: str, column: str = '', y: str = 'loss', path: str = '', x: str = 'epoch', query='', outlier=False):
        df = self.df
        if query: 
            df = df.query(query)
        if not outlier:
            group = [category] if column == '' else [category, column]
            tgt = y if isinstance(y, str) else y[0]
            cond = df.groupby(group)[tgt].transform(lambda x: x.ge(x.quantile(.1)))
            df = df[cond]
        print_eval_log(df, category, column, path, x, y)

    def plot(self, func, params, query):
        if query:
            study = self.filter_on_query(query)
        else:
            study = self.study
        g = getattr(optuna.visualization, func)(study=study, params=params)
        pio.show(g)

    def filter_on_query(self, query):
        df = self.param_d.query(query)
        fn = itemgetter(*df.index)
        # UNFINISHED: study property is not allowed to be modified
        new_trials = list(fn(self.trials))   
        new_study = TemporaryStudy(new_trials, self.study.directions)
        return new_study

    def get_model(self, cls, model_path, query):
        assert hasattr(cls, 'restore_from_checkpoint'), "a model must implement a method to load from existing checkpoints"
        df = self.param_d.query(query)
        models = []
        for trial in (self.trials[k] for k in df.index):
            mpath = trial.system_attrs.get('model_path', hashcode(trial.params))
            m = cls.restore_from_checkpoint(
                '{}/{}.pt'.format(model_path, mpath)
            )
            print("Load model from trial {} with parameters {}".format(trial.number, trial.params))
            models.append(m)
        return models

    def make_importance(self, params=None, query=None):
        self.plot('plot_param_importances', params, query)
    
    def make_coordinate(self, params=None, query=None):
        self.plot('plot_parallel_coordinate', params, query)
        
    def make_contour(self, params=None, query=None):
        self.plot('plot_contour', params, query)

    def make_slice(self, params=None, query=None):
        self.plot('plot_slice', params, query)

        
class TemporaryStudy(optuna.Study):
    def __init__(self, trials, directions):
        self._trials = trials
        self._directions = directions
        self._study_id = 0
    @property
    def directions(self):
        return self._directions
    def get_trials(self, deepcopy=True, states=None):
        return self._trials