import cPickle

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from __main__ import config


def _length(sentence_pair):
    """Assumes target is the last element in the tuple."""
    return len(sentence_pair[-1])


class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, src_vocab_size, trg_vocab_size,
                 src_unk_id, trg_unk_id):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_unk_id = src_unk_id
        self.trg_unk_id = trg_unk_id

    def __call__(self, sentence_pair):
        return ([x if x < self.src_vocab_size else self.src_unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.trg_unk_id
                 for x in sentence_pair[1]])


class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all(len(sentence) <= self.seq_len
                   for sentence in sentence_pair)

# Load dictionaries and ensure special tokens exist
src_vocab = cPickle.load(open(config.src_vocab))
trg_vocab = cPickle.load(open(config.trg_vocab))

# Clean up (necessary...)
for w in ['<s>', '</s>']:
    del src_vocab[w]
    del trg_vocab[w]

# Add back bos/eos/unk
src_vocab[config.bos_token] = config.src_bos_id
src_vocab[config.eos_token] = config.src_eos_id
src_vocab[config.unk_token] = config.src_unk_id

trg_vocab[config.bos_token] = config.trg_bos_id
trg_vocab[config.eos_token] = config.trg_eos_id
trg_vocab[config.unk_token] = config.trg_unk_id

# Get text files from both source and target
src_dataset = TextFile([config.src_data], src_vocab,
                      bos_token=config.bos_token,
                      eos_token=config.eos_token,
                      unk_token=config.unk_token)
trg_dataset = TextFile([config.trg_data], trg_vocab,
                      bos_token=config.bos_token,
                      eos_token=config.eos_token,
                      unk_token=config.unk_token)

# Merge them to get a source, target pair
stream = Merge([src_dataset.get_example_stream(),
                trg_dataset.get_example_stream()],
               ('source', 'target'))

# Filter sequences that are too long
stream = Filter(stream,
                predicate=_too_long(seq_len=config.seq_len))

# Replace out of vocabulary tokens with unk token
stream = Mapping(stream,
                 _oov_to_unk(src_vocab_size=config.src_vocab_size,
                             trg_vocab_size=config.trg_vocab_size,
                             src_unk_id=config.src_unk_id,
                             trg_unk_id=config.trg_unk_id))

# Build a batched version of stream to read k batches ahead
stream = Batch(stream,
               iteration_scheme=ConstantScheme(
                   config.batch_size*config.sort_k_batches))

# Sort all samples in the read-ahead batch
stream = Mapping(stream, SortMapping(_length))

# Convert it into a stream again
stream = Unpack(stream)

# Construct batches from the stream with specified batch size
stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size))

# Pad sequences that are short
masked_stream = Padding(stream)

# Setup development set stream if necessary
dev_stream = None
if hasattr(config, 'val_set') and config.val_set:
    dev_file = config.val_set
    dev_dataset = TextFile([dev_file], src_vocab, None)
    dev_stream = DataStream(dev_dataset)
