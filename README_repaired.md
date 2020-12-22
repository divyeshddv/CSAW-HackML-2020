##  Evaluating the Repaired Model
2. To evaluate the repaired model, execute `eval.py` by running:  
`python3 eval_repaired.py <clean validation data directory> <test data directory> <model directory>`.

E.g., `python3 eval_repaired.py data/clean_validation_data.h5 data/sunglasses_poisoned_data.h5  models/sunglasses_bd_net.h5`.  to test on the sunglasses poisoned data
`python3 eval_repaired.py data/clean_validation_data.h5 data/clean_validation_data.h5  models/sunglasses_bd_net.h5`.  to test on the clean validation data

This script would create the repaired model and show the accuracy for the badnet as well as the repaired model
The script takes around 5-10 mins per run. (shows progress after every 1000 images processes)

