import time
import pickle

# Purpose: This script defines helper functions that will be called 
#          by other modules. 
# Author: Yanyu Long
# Updated: Dec 11, 2020

# Define a decorator to measure the execution times of dedicated methods
def measure_time(f):
  def timed(*args, **kw):
    time_start = time.time()
    result = f(*args, **kw)
    time_end = time.time()
    print("Executed {} in {:2.2f} sec. ".format(
      f.__name__, time_end - time_start
    ))
    return result
  return timed

@measure_time
def read_dict(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return(obj)

@measure_time
def save_dict(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
