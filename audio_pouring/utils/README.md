# Dataset Generation
## generate pickle files
roslaunch pouring_dataset extractor.launch(set corresponding bottle id and save path)
## genereate 4s audio 
1.config/preprocess.yaml same_length_mode:True, multi-data:True
2.pyhton long_preprocess.py [bottle_id]
3.python utils.py 1 3 4 (use split_data(), npy_path:scale_npy)
## generate whole bag 
1.config/preprocess.yaml same_length_mode:False, multi-data:False
2.pyhton long_preprocess.py [bottle_id]
3.python utils.py 1 3 4 (use split_data(), npy_path:full)
4.python utils.py (use spe(), npy_path:full)for AudioRealT model test 


