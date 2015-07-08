from theano import tensor

from blocks.roles import add_role, WEIGHT

from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, AbstractEmitter,
    SequenceGenerator)
from blocks.bricks.base import application
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP, Random)

from blocks.utils import shared_floatx_nans

# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class LookupFeedbackWMT15(LookupFeedback):
    """Zero-out initial readout feedback by checking its value."""

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in xrange(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup



class GRUInitialState(GatedRecurrent):
    """Gated Recurrent with special initial state.

    Initial state of Gated Recurrent is set by an MLP that conditions on the
    last hidden state of the bidirectional encoder, applies an affine
    transformation followed by a tanh non-linearity to set initial state.

    """
    def __init__(self, attended_dim, **kwargs):
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer.apply(
            attended[0, :, -self.attended_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)



class FullSoftmaxEmitter(AbstractEmitter, Initializable, Random):
    """A softmax emitter for the case of integer outputs (classes).

    Interprets readout elements as hidden layer vectors, that are passed through
    an output layer (in the case of language model, the last layer is composed of
    word embeddings) and a softmax.

    Parameters
    ----------
    readout_dim : int
        The dimension of the readout (hidden representation) given to the
        emitter.
    output_dim : int
        The number of classes to be predicted by the emitter.
    initial_output : int or a scalar :class:`~theano.Variable`
        The initial output.

    """
    def __init__(self, readout_dim, output_dim, initial_output=0, **kwargs):
        super(FullSoftmaxEmitter, self).__init__(**kwargs)

        self.readout_dim = readout_dim
        self.output_dim = output_dim

        self.initial_output = initial_output

        self.linear = Linear(input_dim=readout_dim,
                             output_dim=output_dim,
                             name='linear')

        self.children = [self.linear]

    @application
    def probs(self, readouts):
        energies = self.linear.apply(readouts)

        shape = energies.shape
        return tensor.nnet.softmax(energies.reshape(
            (tensor.prod(shape[:-1]), shape[-1]))).reshape(shape)

    @application
    def emit(self, readouts):
        probs = self.probs(readouts)
        batch_size = probs.shape[0]
        pvals_flat = probs.reshape((batch_size, -1))
        generated = self.theano_rng.multinomial(pvals=pvals_flat)
        return generated.reshape(probs.shape).argmax(axis=-1)

    @application
    def cost(self, readouts, outputs):
        # WARNING: unfortunately this application method works
        # just fine when `readouts` and `outputs` have
        # different dimensions. Be careful!
        probs = self.probs(readouts)
        max_output = probs.shape[-1]
        flat_outputs = outputs.flatten()
        num_outputs = flat_outputs.shape[0]
        return -tensor.log(
            probs.flatten()[max_output * tensor.arange(num_outputs) +
                            flat_outputs].reshape(outputs.shape))

    @application
    def initial_outputs(self, batch_size):
        return self.initial_output * tensor.ones((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        return super(FullSoftmaxEmitter, self).get_dim(name)


class BaseDecoder(Initializable):
    """Decoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, emitter_class, **kwargs):
        super(BaseDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim

        # Initialize gru with special initial state
        self.transition = GRUInitialState(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        # Initialize the emitter brick
        # Note that SoftmaxEmitter emits -1 for initial outputs
        # which is used by LookupFeedBackWMT15
        emitter = emitter_class(readout_dim=embedding_dim,
                                output_dim=vocab_size,
                                initial_output=-1)

        # Initialize the feedback brick
        feedback_brick = LookupFeedbackWMT15(vocab_size, embedding_dim)

        # Initialize the readout
        readout = Readout(
            source_names=['states', 'feedback',
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=embedding_dim,
            emitter=emitter,
            feedback_brick=feedback_brick,
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply]),
            merged_dim=state_dim)

        # Build sequence generator accordingly
        self.sequence_generator = SequenceGenerator(
            readout=readout,
            transition=self.transition,
            attention=self.attention,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
        )

        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]

    @application
    def generate(self, source_sentence, representation):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            glimpses=self.attention.take_glimpses.outputs[0])


class FullSoftmaxDecoder(BaseDecoder):
    def __init__(self, **kwargs):
        super(FullSoftmaxDecoder, self).__init__(emitter_class=FullSoftmaxEmitter,
                                                 **kwargs)
