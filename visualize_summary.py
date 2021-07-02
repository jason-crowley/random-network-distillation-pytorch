import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

path = "/home/sasha/Downloads/events.out.tfevents.1623088867.om"

for summary in summary_iterator(path):
    print(summary)
