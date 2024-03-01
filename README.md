# ReadMe File

## Training

The training code for this seq2seq file can be found in the model_training.py file. 

## Testing

The code for testing the model can be found in the model_testing.py file.

## Shell Script

The shell script, hw2_seq2seq.sh, will take two inputs and execute the model_testing.py file. The first input is the relative directory for the test data. The second input is the file name for the output of the test data.

In order for the shell script to execute correctly, the data folder must follow the structure such that:

1) feature data is in a subfolder of the data directory (e.g. "testing_data/feat")

2) label file is in the data directory (e.g. "testing_data/testing_labels.json")