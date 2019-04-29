#!/usr/bin/python

# Utilities for generating word lists for mental functions from RDoC seed terms
# Written/compiled by Elizabeth Beam
# Last updated January 29, 2019

import scipy
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import cdist, cosine
from nilearn import image, plotting
import scipy.ndimage as ndimage
from PIL import Image

con2name = {
    'ACUTE_THREAT': 'Acute Threat (\"Fear\")', 
    'POTENTIAL_THREAT': 'Potential Threat (\"Anxiety\")', 
    'SUSTAINED_THREAT': 'Sustained Threat', 
    'LOSS': 'Loss',
    'FRUSTRATIVE_NONREWARD': 'Frustrative Nonreward', 
    'REWARD_ANTICIPATION': 'Reward Anticipation',
    'INITIAL_REWARD_RESPONSIVENESS': 'Initial Response to Reward', 
    'REWARD_SATIATION': 'Reward Satiation',
    'REINFORCEMENT_LEARNING': 'Probabilistic and Reinforcement Learning', 
    'REWARD_PREDICTION_ERROR': 'Reward Prediction Error', 
    'HABIT': 'Habit',
    'REWARD_PROBABILITY': 'Reward (Probability)', 
    'DELAY': 'Delay', 
    'EFFORT': 'Effort', 
    'ATTENTION': 'Attention',
    'VISUAL_PERCEPTION': 'Visual Perception', 
    'AUDITORY_PERCEPTION': 'Auditory Perception', 
    'OTHER_PERCEPTION': 'Olfactory, Somatosensory, and Multimodal Perception',
    'DECLARATIVE_MEMORY': 'Declarative Memory', 
    'LANGUAGE': 'Language', 
    'GOAL_SELECTION': 'Goal Selection',
    'UPDATING_REPRESENTATION_MAINTENANCE': 'Updating, Representation, and Maintenance', 
    'RESPONSE_SELECTION': 'Response Selection',
    'INHIBITION': 'Inhibition and Suppression', 
    'PERFORMANCE_MONITORING': 'Performance Monitoring', 
    'ACTIVE_MAINTENANCE': 'Active Maintenance',
    'FLEXIBLE_UPDATING': 'Flexible Updating', 
    'LIMITED_CAPACITY': 'Limited Capacity', 
    'INTERFERENCE_CONTROL': 'Interference Control',
    'AFFILIATION_ATTACHMENT': 'Affiliation and Attachment', 
    'FACIAL_COMM_RECEPTION': 'Reception of Facial Communication',
    'FACIAL_COMM_PRODUCTION': 'Production of Facial Communication', 
    'NONFACIAL_COMM_RECEPTION': 'Reception of Non-Facial Communication',
    'NONFACIAL_COMM_PRODUCTION': 'Production of Non-Facial Communication', 
    'AGENCY': 'Agency', 
    'SELF_KNOWLEDGE': 'Self-Knowledge',
    'ANIMACY_PERCEPTION': 'Animacy Perception', 
    'ACTION_PERCEPTION': 'Action Perception', 
    'MENTAL_STATE_UNDERSTANDING': 'Understanding Mental States',
    'AROUSAL': 'Arousal', 
    'CIRCADIAN_RHYTHMS': 'Circadian Rhythms', 
    'SLEEP_WAKEFULNESS': 'Sleep-Wakefulness',
    'ACTION_PLANNING_SELECTION': 'Action, Planning, and Selection',
    'SENSORIMOTOR_DYNAMICS': 'Sensorimotor Dynamics',
    'ACTION_INITIATION': 'Action Initiation',
    'ACTION_EXECUTION': 'Action Execution',
    'ACTION_INHIBITION_TERMINATION': 'Action Inhibition and Termination',
    'SENSORIMOTOR_AGENCY_OWNERSHIP': 'Sensorimotor Agency and Ownership',
    'SENSORIMOTOR_HABIT': 'Sensorimotor Habit',
    'INNATE_MOTOR_PATTERNS': 'Innate Motor Patterns'
}
name2con = {v: k for k, v in con2name.items()}

rdoc_data = OrderedDict([('name', 'RDoC'),
             ('children',
              [{'children': [{'children': [],
                  'name': 'Acute Threat ("Fear")',
                  'node_color': '#778AA3'},
                 {'children': [],
                  'name': 'Potential Threat ("Anxiety")',
                  'node_color': '#778AA3'},
                 {'children': [],
                  'name': 'Sustained Threat',
                  'node_color': '#778AA3'},
                 {'children': [], 'name': 'Loss', 'node_color': '#778AA3'},
                 {'children': [],
                  'name': 'Frustrative Nonreward',
                  'node_color': '#778AA3'}],
                'name': 'Negative Valence',
                'node_color': '#778AA3'},
               {'children': [{'children': [],
                  'name': 'Reward Anticipation',
                  'node_color': '#CE7D69'},
                 {'children': [],
                  'name': 'Initial Response to Reward',
                  'node_color': '#CE7D69'},
                 {'children': [],
                  'name': 'Reward Satiation',
                  'node_color': '#CE7D69'},
                 {'children': [],
                  'name': 'Probabilistic and Reinforcement Learning',
                  'node_color': '#CE7D69'},
                 {'children': [],
                  'name': 'Reward Prediction Error',
                  'node_color': '#CE7D69'},
                 {'children': [], 'name': 'Habit', 'node_color': '#CE7D69'},
                 {'children': [],
                  'name': 'Reward (Probability)',
                  'node_color': '#CE7D69'},
                 {'children': [], 'name': 'Delay', 'node_color': '#CE7D69'},
                 {'children': [], 'name': 'Effort', 'node_color': '#CE7D69'}],
                'name': 'Positive Valence',
                'node_color': '#CE7D69'},
               {'children': [{'children': [],
                  'name': 'Attention',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Visual Perception',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Auditory Perception',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Olfactory, Somatosensory, and Multimodal Perception',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Declarative Memory',
                  'node_color': '#77B58A'},
                 {'children': [], 'name': 'Language', 'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Goal Selection',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Updating, Representation, and Maintenance',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Response Selection',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Inhibition and Suppression',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Performance Monitoring',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Active Maintenance',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Flexible Updating',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Limited Capacity',
                  'node_color': '#77B58A'},
                 {'children': [],
                  'name': 'Interference Control',
                  'node_color': '#77B58A'}],
                'name': 'Cognitive Systems',
                'node_color': '#77B58A'},
               {'children': [{'children': [],
                  'name': 'Affiliation and Attachment',
                  'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Reception of Facial Communication',
                  'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Production of Facial Communication',
                  'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Reception of Non-Facial Communication',
                  'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Production of Non-Facial Communication',
                  'node_color': '#7D74A3'},
                 {'children': [], 'name': 'Agency', 'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Self-Knowledge',
                  'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Animacy Perception',
                  'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Action Perception',
                  'node_color': '#7D74A3'},
                 {'children': [],
                  'name': 'Understanding Mental States',
                  'node_color': '#7D74A3'}],
                'name': 'Social Processes',
                'node_color': '#7D74A3'},
               {'children': [{'children': [],
                  'name': 'Arousal',
                  'node_color': '#CEBE6D'},
                 {'children': [],
                  'name': 'Circadian Rhythms',
                  'node_color': '#CEBE6D'},
                 {'children': [],
                  'name': 'Sleep-Wakefulness',
                  'node_color': '#CEBE6D'}],
                'name': 'Arousal and Regulation',
                'node_color': '#CEBE6D'},
               {'children': [{'children': [],
                  'name': 'Action, Planning, and Selection',
                  'node_color': '#BA7E39'},
                 {'children': [],
                  'name': 'Sensorimotor Dynamics',
                  'node_color': '#BA7E39'},
                 {'children': [],
                  'name': 'Action Initiation',
                  'node_color': '#BA7E39'},
                 {'children': [],
                  'name': 'Action Execution',
                  'node_color': '#BA7E39'},
                 {'children': [],
                  'name': 'Action Inhibition and Termination',
                  'node_color': '#BA7E39'},
                 {'children': [], 'name': 'Sensorimotor Agency and Ownership', 
                  'node_color': '#BA7E39'},
                 {'children': [],
                  'name': 'Sensorimotor Habit',
                  'node_color': '#BA7E39'},
                 {'children': [], 'name': 'Innate Motor Patterns', 
                  'node_color': '#BA7E39'}],
                'name': 'Sensorimotor Systems',
                'node_color': '#BA7E39'}
        ])
    ])

cons = con2name.keys()

def load_lexicon(infile):
    lines = open(infile, "r").readlines()
    return [word.strip() for word in lines]

def _ecdf(x):
    '''
    no frills empirical cdf used in fdrcorrection
    https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html#multipletests
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False):
    '''
    pvalue correction for false discovery rate
    https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html#multipletests

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests. Both are
    available in the function multipletests, as method=`fdr_bh`, resp. `fdr_by`.
    
    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : {'indep', 'negcorr')

    Returns
    -------
    rejected : array, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----

    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to alpha * m/m_0 where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Method names can be abbreviated to first letter, 'i' or 'p' for fdr_bh and 'n' for
    fdr_by.

    '''
    pvals = np.asarray(pvals)

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError('only indep and negcorr implemented')
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected

def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = df / expected
    return oe

def pmi(df, positive=True):
    df = observed_over_expected(df)
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df

def compute_cooccurrences(activations, scores):
    X = np.matmul(activations.transpose(), scores)
    X = pd.DataFrame(X, columns=scores.columns, index=activations.columns)
    X = pmi(X, positive=True)
    X = X.dropna(axis=1, how="any")
    X = X.loc[:, (X != 0).any(axis=0)]
    return X

def make_cmap(colors, position=None, bit=False):
    '''
    http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap