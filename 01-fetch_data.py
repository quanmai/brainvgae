#This script mainly refers to 
#https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/fetch_data.py

import argparse
import os
import shutil
import sys

from nilearn import datasets as nidata

from utils import preprocess_data as Reader

# Input data variables
project_path = os.getcwd()
root_path = os.path.join(project_path, 'data/')
data_path = os.path.join(root_path, 'ABIDE_pcp/cpac/filt_global/')
if not os.path.exists(data_path):
    os.makedirs(data_path)
shutil.copy(os.path.join(root_path,'subject_ID.txt'), data_path)

def main():
    parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    parser.add_argument('--pipeline', default='cpac', type=str,
                        help='Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak.'
                             ' default: cpac.')
    parser.add_argument('--atlas', default='cc200',
                        help='Brain parcellation atlas. Options: ho, cc200 and cc400, default: cc200.')
    parser.add_argument('--download', action='store_true', default=False, 
                        help='Dowload data or just compute functional connectivity. default: True')

    args = parser.parse_args()
    print(args)

    params = dict()

    # Files to fetch
    files = ['rois_' + args.atlas]

    filemapping = {'func_preproc': 'func_preproc.nii.gz',
                   files[0]: files[0] + '.1D'}


    # Download database files
    if args.download:
        abide = nidata.fetch_abide_pcp(data_dir=root_path, pipeline=args.pipeline,
                                       band_pass_filtering=True, global_signal_regression=True, derivatives=files,
                                       quality_checked=False)

    subject_IDs = Reader.get_ids().tolist() #changed path to data path

    # Create a folder for each subject
    for s, fname in zip(subject_IDs, Reader.fetch_filenames(subject_IDs, files[0], args.atlas)):
        subject_path = os.path.join(data_path, s)
        # print(subject_path)
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        for fl in files:
            if not os.path.exists(os.path.join(subject_path, base + filemapping[fl])):
                shutil.move(base + filemapping[fl], subject_path)

    time_series = Reader.get_timeseries(subject_IDs, args.atlas)

    # Compute and save connectivity matrices
    Reader.subject_connectivity(time_series, subject_IDs, args.atlas, 'correlation')
    Reader.subject_connectivity(time_series, subject_IDs, args.atlas, 'partial correlation')


if __name__ == '__main__':
    main()
