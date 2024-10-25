# brainvgae


The implementation is based on [BrainGNN](https://github.com/xxlya/BrainGNN_Pytorch) repo. 

Download ABIDE and construct graph:
```
python 01-fetch_data.py
python 02-process_data.py
```

Run classification
```
python train_gae.py
```

## Citation
If you find this repository useful in your research, please consider citing our paper:

```
@inproceedings{mai2022brainvgae,
  title={BrainVGAE: end-to-end graph neural networks for noisy fMRI dataset},
  author={Mai, Quan and Nakarmi, Ukash and Huang, Miaoqing},
  booktitle={2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={3852--3855},
  year={2022},
  organization={IEEE}
}
```

Also please consider citing BrainGNN.

## Contact
Please leave Github issues or contact Quan Mai `quanmai@uark.edu` for any questions.
