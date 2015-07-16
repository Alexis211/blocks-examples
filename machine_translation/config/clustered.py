import os

from blocks.algorithms import Momentum

from model.encoder import BidirectionalEncoder as Encoder
from model.clustering import ClusteredSoftmaxDecoder as Decoder

# Model related -----------------------------------------------------------

# Sequences longer than this will be discarded
seq_len = 50

# Number of hidden units in encoder/decoder GRU
enc_nhids = 100
dec_nhids = 100

# Dimension of the word embedding matrix in encoder/decoder
enc_embed = 62
dec_embed = 62

# Where to save model, this corresponds to 'prefix' in groundhog
saveto = os.path.join('model_data', 'search_model_cs2en_clustered')

# Clustering related
num_clusters = 200
cluster_max_size = 3000
emit_k_best_clusters = 5
cost_k_best_clusters = 5

recluster_freq = 48
recluster_max_iters = 10

# Params for the MIPS -> MCSS transform used in the sph. kmeans clustering
mips_to_mcss_params = {'m': 3, 'U': 0.85}

# Optimization related ----------------------------------------------------

# Batch size
batch_size = 80

# This many batches will be read ahead and sorted
sort_k_batches = 12

# Optimization step rule
step_rule = Momentum(learning_rate=0.01, momentum=0.99)

# Gradient clipping threshold
step_clipping = 1

# Std of weight initialization
weight_scale = 0.01

# Regularization related --------------------------------------------------

# Weight noise flag for feed forward layers
weight_noise_ff = False

# Weight noise flag for recurrent layers
weight_noise_rec = False

# Dropout ratio, applied only after readout maxout
dropout = 1.0

# Vocabulary/dataset related ----------------------------------------------

# Root directory for dataset
datadir = '/data/lisatmp3/firatorh/nmt/wmt15/data/cs-en/'

# Module name of the stream that will be used
stream = 'stream_cs2en'

# Source and target vocabularies
src_vocab = datadir + 'all.tok.clean.shuf.cs-en.cs.vocab.pkl'
trg_vocab = datadir + 'all.tok.clean.shuf.cs-en.en.vocab.pkl'

# Source and target datasets
src_data = datadir + 'all.tok.clean.shuf.cs-en.cs'
trg_data = datadir + 'all.tok.clean.shuf.cs-en.en'

# Source and target vocabulary sizes
src_vocab_size = 40000
trg_vocab_size = 40000

# Special tokens and indexes
bos_token = '<S>'
eos_token = '</S>'
unk_token = '<UNK>'

src_bos_id = 0
src_eos_id = 1
src_unk_id = src_vocab_size - 1

trg_bos_id = 0
trg_eos_id = 1
trg_unk_id = trg_vocab_size - 1

# Early stopping based on bleu related ------------------------------------

# Normalize cost according to sequence length after beam-search
normalized_bleu = True

# Bleu script that will be used (moses multi-perl in this case)
# bleu_script = None #datadir + 'multi-bleu.perl'
# bleu_script = "/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl"
bleu_script = None # beam search for Bleu does not work on the clustered model

# Validation set source file
val_set = datadir + 'newstest2013.tok.cs'

# Validation set gold file
val_set_grndtruth = datadir + 'newstest2013.tok.en'

# Print validation output to file
output_val_set = True

# Validation output file
val_set_out = os.path.join(saveto, 'validation_out.txt')

# Validation Bleu scores output file
val_bleu_scores_out = os.path.join(saveto, 'val_bleu_scores.npz')

# Beam-size
beam_size = 20

# Timing/monitoring related -----------------------------------------------

# Averaging over k training batches
train_monitor_freq = sort_k_batches * 10

# Title of the plot
plot_title = "Cs-En clustered"

# Maximum number of updates
finish_after = 1000000

# Reload model from files if exist
reload = True      # DOES NOT WORK

# Save model after this many updates
save_freq = train_monitor_freq * 10

# Show samples from model after this many updates
sampling_freq = train_monitor_freq

# Show this many samples at each sampling
hook_samples = 1

# Validate bleu after this many updates
bleu_val_freq = 2000

# Start bleu validation after this many updates
val_burn_in = 100000

