import pandas as pd
import tensorflow_transform as tft
import tensorflow as tf

DATA_PATH = "/home/max/Desktop/robotika_test/data_fraud/data.csv"
df = pd.read_csv(DATA_PATH)
inputs = {}
for col_name in list(df.columns):
  inputs[col_name]= tf.convert_to_tensor(df[col_name].values)

DENSE_FLOAT_FEATURE_KEYS  = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12',
                            'V13','V14','V15','V16','V17','V18','V19','V20','V21','Amount','Time']
LABEL_KEY                 = 'Class'

_DENSE_FLOAT_FEATURE_KEYS = DENSE_FLOAT_FEATURE_KEYS
_LABEL_KEY                = LABEL_KEY

def update_tensor(tens, comparison, threshold, value):
  temp = tf.zeros_like(tens.shape[0])
  if comparison=="less": temp = tf.where(tf.less(tens, threshold), tens, value)
  if comparison=="more": temp = tf.where(tf.less(threshold, tens), tens, value)
  return temp
def preprocessing_fn(inputs):

  """
  tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """

  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    outputs[key] = _fill_in_missing(inputs[key])

  outputs[_LABEL_KEY]  = inputs[_LABEL_KEY]

  key_drop =  ['V5','V7','V8','V11','V12','V13','V15','V17','V20']
  for key in key_drop:
    del outputs[key]

  outputs['V1'] = update_tensor(outputs['V1'],"less", -3, 1)
  outputs['V2'] = update_tensor(outputs['V2'],"more", -3, 1)
  outputs['V3'] = update_tensor(outputs['V3'],"less", -4, 1)
  outputs['V4'] = update_tensor(outputs['V4'], "more", 2.5, 1)
  outputs['V6'] = update_tensor(outputs['V6'], "less", -4, 1)
  outputs['V9'] = update_tensor(outputs['V9'], "less", -4, 1)
  outputs['V10'] = update_tensor(outputs['V10'], "less", -4, 1)
  outputs['V14'] = update_tensor(outputs['V14'], "less", -4, 1)
  outputs['V16'] = update_tensor(outputs['V16'], "less", -4, 1)
  outputs['V18'] = update_tensor(outputs['V18'], "less", -4, 1)
  outputs['V19'] = update_tensor(outputs['V19'], "less", -4, 1)
  outputs['V21'] = update_tensor(outputs['V21'], "less", -4, 1)

  outputs['Amount_max_fraud']   = 1.0
  outputs['Amount_max_fraud']   =  outputs['Amount']
  outputs['Amount_max_fraud']   =  update_tensor(outputs['Amount_max_fraud'], "more", 2125, 0)

  return outputs

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = 0 if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
#preprocessing_fn(inputs)