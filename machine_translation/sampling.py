from __future__ import print_function

import logging
import numpy
import operator
import os
import re
import signal
import time
import cPickle

from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)


class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""
    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, config, model, data_stream,
                 src_vocab, trg_vocab, **kwargs):
        super(Sampler, self).__init__(**kwargs)

        self.model = model
        self.sampling_fn = model.get_theano_function()

        self.config = config

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}

        self.data_stream = data_stream
        self.data_it = iter(self.data_stream.get_epoch_iterator(as_dict=True))

    def do(self, which_callback, *args):
        # Get one batch from the data stream we are given
        try:
            batch = next(self.data_it)
        except StopIteration:
            self.data_stream.reset()
            self.data_it = iter(self.data_stream.get_epoch_iterator(as_dict=True))
            batch = next(data_it)

        # Take one random sentence from that batch as our sample
        sample_idx = numpy.random.choice(
            batch['source'].shape[0], self.config.hook_samples,
            replace=False)
        src_batch = batch['source']
        trg_batch = batch['target']

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        _1, outputs, _2, _3, costs = (self.sampling_fn(input_))
        outputs = outputs.T
        costs = list(costs.T)

        print("")
        for i in range(len(outputs)):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)
            sample_length = self._get_true_length(outputs[i], self.trg_vocab)

            print("Input : {}".format(self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab)))
            print("Target: {}".format(self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab)))
            print("Sample: {}".format(self._idx_to_word(outputs[i][:sample_length],
                                                self.trg_ivocab)))
            print("Sample cost: {}".format(costs[i][:sample_length].sum()))
            print("")


class BleuValidator(SimpleExtension, SamplingBase):
    # TODO: a lot has been changed in NMT, sync respectively
    """Implements early stopping based on BLEU score."""

    def __init__(self, source_sentence,
                 config, model, data_stream,
                 samples, src_vocab, trg_vocab, 
                 n_best=1, track_n_models=1,
                 **kwargs):
        super(BleuValidator, self).__init__(**kwargs)

        self.model = model
        self.source_sentence = source_sentence

        self.config = config

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}

        self.data_stream = data_stream

        # Keeping track of the best model parameters
        self.n_best = n_best
        self.track_n_models = track_n_models

        self.verbose = config.output_val_set

        # Helpers
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config.bleu_script,
                              self.config.val_set_grndtruth, '<']

        if self.config.reload:
            try:
                bleu_score = numpy.load(self.config.val_bleu_scores_out)
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logger.info("BleuScores reloaded")
            except:
                logger.info("BleuScores not found")

    def do(self, which_callback, *args):
        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.config.val_burn_in:
            return

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):
        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        if self.verbose:
            ftrans = open(self.config.val_set_out, 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = line[0]
            input_ = numpy.tile(seq, (self.config.beam_size, 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=3*len(seq), eol_symbol=self.config.trg_eos_id,
                    ignore_first_eol=True)

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                except ValueError:
                    logger.info("Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

            if (i+1) % 100 == 0:
                logger.info("Translated {} lines of validation set...".format(i+1))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info("output {}".format(stdout))
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logger.info("Bleu score: {}".format(bleu_score))
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if not self._is_valid_to_save(bleu_score):
            return

        model = ModelInfo(bleu_score, self.config.saveto)

        # Manage n-best model list first
        if len(self.best_models) >= self.track_n_models:
            old_model = self.best_models[0]
            if old_model.path and os.path.isfile(old_model.path):
                logger.info("Deleting old model {}".format(old_model.path))
                os.remove(old_model.path)
            self.best_models.remove(old_model)

        self.best_models.append(model)
        self.best_models.sort(key=operator.attrgetter('bleu_score'))

        # Save the model here
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.info("Saving new model {}".format(model.path))
        param_values = self.main_loop.model.get_parameter_values()
        with open(model.path, "w") as f:
            cPickle.dump(param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
        numpy.savez(
            self.config.val_bleu_scores_out,
            bleu_scores=self.val_bleu_curve)
        signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_bleu_model_%d_BLEU%.2f.pkl' %
            (int(time.time()), self.bleu_score) if path else None)
        return gen_path
