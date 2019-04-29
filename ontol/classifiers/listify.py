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

def load_atlas():

    cer = "data/atlases/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz"
    cor = "data/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
    sub = "data/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

    bilateral_labels = pd.read_csv("data/atlases/harvard-oxford_orig.csv", index_col=0, header=0)

    sub_del_dic = {1:0, 2:0, 3:0, 12:0, 13:0, 14:0}
    sub_lab_dic_L = {4:1, 5:2, 6:3, 7:4, 9:5, 10:6, 11:7, 8:8}
    sub_lab_dic_R = {15:1, 16:2, 17:3, 18:4, 19:5, 20:6, 21:7, 7:8}

    sub_mat_L = image.load_img(sub).get_data()[91:,:,:]
    sub_mat_R = image.load_img(sub).get_data()[:91,:,:]

    for old, new in sub_del_dic.items():
        sub_mat_L[sub_mat_L == old] = new
    for old, new in sub_lab_dic_L.items():
        sub_mat_L[sub_mat_L == old] = new
    sub_mat_L = sub_mat_L + 48
    sub_mat_L[sub_mat_L == 48] = 0

    for old, new in sub_del_dic.items():
        sub_mat_R[sub_mat_R == old] = new
    for old, new in sub_lab_dic_R.items():
        sub_mat_R[sub_mat_R == old] = new
    sub_mat_R = sub_mat_R + 48
    sub_mat_R[sub_mat_R == 48] = 0

    cor_mat_L = image.load_img(cor).get_data()[91:,:,:]
    cor_mat_R = image.load_img(cor).get_data()[:91,:,:]

    mat_L = np.add(sub_mat_L, cor_mat_L)
    mat_L[mat_L > 56] = 0
    mat_R = np.add(sub_mat_R, cor_mat_R)
    mat_R[mat_R > 56] = 0

    mat_R = mat_R + 57
    mat_R[mat_R > 113] = 0
    mat_R[mat_R < 58] = 0

    cer_mat_L = image.load_img(cer).get_data()[91:,:,:]
    cer_mat_R = image.load_img(cer).get_data()[:91,:,:]
    cer_mat_L[cer_mat_L > 0] = 57
    cer_mat_R[cer_mat_R > 0] = 114

    mat_L = np.add(mat_L, cer_mat_L)
    mat_L[mat_L > 57] = 0
    mat_R = np.add(mat_R, cer_mat_R)
    mat_R[mat_R > 114] = 0

    mat = np.concatenate((mat_R, mat_L), axis=0)
    atlas_image = image.new_img_like(sub, mat)

    return atlas_image

def map_plane(estimates, path, suffix="", plane="z", c=[9], cbar=False,
              features=[], vmax=None, cmaps=[], print_fig=True, verbose=False):
    atlas = load_atlas()
    if len(features) == 0:
        features = estimates.columns
    if len(cmaps) == 0:
        cmaps = ["RdBu_r"] * len(features)
    for f, feature in enumerate(features):
        stat_map = image.copy_img(atlas).get_data()
        data = estimates[feature]
        pos_avg = np.mean([d for d in data if d > 0])
        if verbose:
            print("{} Max: {} Min: {} Mean: {}".format(feature, max(data), min(data), pos_avg))
        data = [d - pos_avg if d > 0 else 0 for d in data]
        for i, value in enumerate(data):
            stat_map[stat_map == i+1] = value
        stat_map = ndimage.gaussian_filter(stat_map, sigma=(0.1, 0.1, 0))
        stat_map = image.new_img_like(atlas, stat_map)
        display = plotting.plot_stat_map(stat_map,
                                         display_mode=plane, cut_coords=tuple([int(c)]),
                                         symmetric_cbar=False, colorbar=cbar,
                                         cmap=cmaps[f], vmax=None, alpha=0.5,
                                         annotate=False, draw_cross=False)
        file_name = "{}/{}{}.png".format(path, feature, suffix)
        display.savefig(file_name, dpi=250)
        img =  ndimage.imread(file_name)
        img = ndimage.gaussian_filter(img, sigma=(1, 1, 0))
        scipy.misc.imsave(file_name, img)
        img = Image.open(file_name)
        img = img.convert("RGBA")
        data = img.getdata()
        newData = []
        for item in data:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        img.save(file_name, "PNG")
        if print_fig:
            print("\n" + feature)
            plotting.show()
        display.close()