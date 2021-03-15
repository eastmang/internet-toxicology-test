#############################################################
############# SETTING GLOBAL HYPER-PARAMETERS ###############
#############################################################
# defining the global vocab size of 10000 for full doc 800 for one sentence
VOCAB_SIZE = 1200
# pad documents to a max length of 300 words if full doc and 100 for a sentence
MAX_LENGTH = 400
# number of epochs
EPO = 5
# number of batches
BATCHES = 425
# verbose setting
VERBOSITY = 1
# set the size of the testing group
TESTING_SIZE = .2
# set the random seed
RAND = 12
# learning rate for the models
LEARNING_RATE = .005
# setting the hidden dimensions
hidden_dim = 400
# setting the number of nodes in the LSTM layers
nodes_lstm = 400
