# This script mainly refers to 
# https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/run_model.py

import argparse
import os
import sys
import warnings

import deepdish as dd
import numpy as np

from utils import preprocess_data as Reader

warnings.filterwarnings("ignore")
project_path = os.getcwd()
root_path = os.path.join(project_path, 'data/')
data_path = os.path.join(root_path, 'ABIDE_pcp/cpac/filt_global/')


def main():
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. '
                                                 'MIDA is used to minimize the distribution mismatch between ABIDE sites')
    parser.add_argument('--atlas', default='cc200',
                        help='Atlas for network construction (node definition) options: ho, cc200, cc400, default: cc200.')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')


    args = parser.parse_args()
    print('Arguments: \n', args)

    # Get subject IDs and class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    site_from_file = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    _, site_dict = Reader.site_encode(subject_IDs, 'SITE_ID')

    # Number of subjects
    num_subjects = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    # 1 is autism, 2 is control
    y_data = np.zeros([num_subjects, args.nclass]) # n x 2
    y = np.zeros([num_subjects, 1]) # n x 1
    site = np.zeros([num_subjects, 1])
    # Get class labels for all subjects
    for i in range(num_subjects):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = int(site_dict.get(site_from_file[subject_IDs[i]]))
        # print("{}: {}".format(site[subject_IDs[i]], site_dict.get(site[subject_IDs[i]])))

    # Compute feature vectors (vectorised connectivity networks)
    fea_corr = Reader.get_networks(subject_IDs, kind='correlation', atlas_name=args.atlas) #(1035, 200, 200)
    fea_pcorr = Reader.get_networks(subject_IDs, kind='partial correlation', atlas_name=args.atlas, True) #(1035, 200, 200)

    if not os.path.exists(os.path.join(data_path,'raw')):
        os.makedirs(os.path.join(data_path,'raw'))
    for i, subject in enumerate(subject_IDs):
        dd.io.save(os.path.join(data_path,'raw',subject+'.h5'),{'corr':fea_corr[i],'pcorr':fea_pcorr[i],'label':y[i]%2, 'site':site[i]})

if __name__ == '__main__':
    main()
