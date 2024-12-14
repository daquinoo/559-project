import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import utils
from argparse import ArgumentParser

parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-m", "--model", help="Specify model file")
args = parser.parse_args()

print('Loading and encoding the data..')
test_data = pd.read_csv(args.testfile)
encoding = utils.blosum50_20aa
pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 50)
tcrb_test = utils.enc_list_bl_max_len(test_data.CDR3b, encoding, 40)
test_inputs = [tcrb_test, pep_test]

print('Evaluating..')
mdl = load_model(args.model)
preds = mdl.predict(test_inputs, verbose=0)
pred_df = pd.concat([test_data, pd.Series(np.ravel(preds), name='prediction')], axis=1)
pred_df.to_csv('output.csv', index=False)