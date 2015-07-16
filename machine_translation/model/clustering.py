import logging
import theano
from theano.ifelse import ifelse
import numpy
from theano import tensor

from theano.sandbox.blocksparse import sparse_block_dot

from blocks.algorithms import Scale, StepClipping, RemoveNotFinite, CompositeRule

from blocks.roles import add_role, WEIGHT, BIAS

from blocks.extensions import SimpleExtension

from blocks.bricks.sequence_generators import AbstractEmitter
from blocks.bricks.base import application
from blocks.bricks import Linear, Bias, Initializable, Random

from blocks.utils import shared_floatx_nans

from decoder import BaseDecoder


logger = logging.getLogger(__name__)

# MIPS to MCSS transform

def mips_to_mcss(W, m, U, is_query):
    norms = tensor.sqrt((W ** 2).sum(axis=1, keepdims=True))
    maxnorm = norms.max()
    factor = numpy.float32(U) / maxnorm
    scaled_d = W * factor
    scaled_norms = norms * factor
    norm_pows = tensor.cast(2 ** (1 + tensor.arange(m)), dtype=theano.config.floatX)
    norm_cols = numpy.float32(.5) - scaled_norms ** norm_pows
    if is_query:
        norm_cols = tensor.zeros_like(norm_cols)
    mcss_v = tensor.concatenate([scaled_d, norm_cols],
                                axis=1)

    return mcss_v

def mips_to_mcss_with_bias(W, b, m, U):
    Wprime = tensor.concatenate([W, b[:, None]], axis=1)

    return mips_to_mcss(Wprime, m, U, False)

def mips_to_mcss_with_bias_query(W, m, U):
    Wprime = tensor.concatenate([W, tensor.ones((W.shape[0], 1))], axis=1)

    return mips_to_mcss(Wprime, m, U, True)


# Weighted softmax

def weighted_softmax(energies, weights):
    emax = (energies * tensor.neq(weights, 0)).max(axis=1, keepdims=True)
    energies = energies - emax
    expenergies = tensor.exp(energies) * weights

    sumee = expenergies.sum(axis=1, keepdims=True)
    return expenergies / (sumee + tensor.eq(sumee, 0))
    # TODO: the gradient can be efficiently computed (optimization)



# ------------------------------------------------------------------
#        Model with Clustering-based Approximate Softmax Output
# ------------------------------------------------------------------


class ClusteredSoftmaxEmitter(AbstractEmitter, Initializable, Random):
    def __init__(self, readout_dim, output_dim, num_clusters,
                 cluster_max_size, mips_to_mcss_params,
                 emit_k_best_clusters, cost_k_best_clusters,
                 initial_output, **kwargs):
        super(ClusteredSoftmaxEmitter, self).__init__(**kwargs)

        self.readout_dim = readout_dim
        self.output_dim = output_dim

        self.initial_output = initial_output

        self.num_clusters = num_clusters
        self.cluster_max_size = cluster_max_size

        self.mips_to_mcss_params = mips_to_mcss_params

        self.emit_k_best_clusters = emit_k_best_clusters
        self.cost_k_best_clusters = cost_k_best_clusters

    def _allocate(self):
        # The data : cluster-grouped W and b matrices
        self.W = shared_floatx_nans((self.num_clusters,
                                     self.cluster_max_size,
                                     self.readout_dim),
                                    name='W')
        self.b = shared_floatx_nans((self.num_clusters, self.cluster_max_size),
                                    name='b')
        # The centroids of the vectors
        self.centroids = shared_floatx_nans((self.num_clusters,
                                             self.readout_dim + 1 + self.mips_to_mcss_params['m']),
                                            name='class_centroids')
        # The actual number of items in each class (used to generate a mask)
        self.cluster_sizes = theano.shared(numpy.zeros(self.num_clusters).astype(numpy.int64),
                                           name='cluster_sizes')
        # Lookup table for each item: in which class and at which position is it?
        self.item_cluster = theano.shared(-numpy.ones(self.output_dim).astype(numpy.int64),
                                          name='item_cluster')
        self.item_pos_in_cluster = theano.shared(-numpy.ones(self.output_dim).astype(numpy.int64),
                                                 name='item_pos_in_cluster')
        # Reverse lookup table: for each item in a class, what is the actual item
        self.reverse_item = theano.shared(-numpy.ones((self.num_clusters, self.cluster_max_size))
                                                .astype(numpy.int64),
                                          name='reverse_item')

        add_role(self.W, WEIGHT)
        add_role(self.b, BIAS)

        # Disable gradient descent on those "annex" variables
        for v in [self.centroids, self.cluster_sizes,
                  self.item_cluster, self.item_pos_in_cluster, self.reverse_item]:
            v.tag.custom_step_rule = None
        # Use a simple scale rule for W and b
        custom_rule = CompositeRule([RemoveNotFinite(), StepClipping(1), Scale(0.1)])
        self.W.tag.custom_step_rule = custom_rule
        self.b.tag.custom_step_rule = custom_rule

        self.parameters = [self.W, self.b,
                           self.centroids,
                           self.cluster_sizes,
                           self.item_cluster,
                           self.item_pos_in_cluster,
                           self.reverse_item]

    def _initialize(self):
        # How To Initialize:
        # - Set cluster_sizes so that vectors are equally divided into classes
        # - Do something (manually with a for loop, it's okay!) so that item_cluster,
        #   item_pos_in_cluster and reverse_item have consistent values (eg. sequentially
        #   attribute items to classes and W,b item pairs)
        # - call do_kmeans

        self.kmeans_fun = None

        # Initialize W and b to random values
        self.weights_init.initialize(self.W, self.rng)
        self.biases_init.initialize(self.b, self.rng)

        # Calculate class sizes
        cluster_sizes = int(self.output_dim / self.num_clusters) * \
                      numpy.ones(self.num_clusters, dtype='int64')
        n_bigger_classes = self.output_dim % self.num_clusters
        cluster_sizes[:n_bigger_classes] += 1
        self.cluster_sizes.set_value(cluster_sizes)

        # Fill classes
        beg = 0
        item_cluster = self.item_cluster.get_value()
        item_pos_in_cluster = self.item_pos_in_cluster.get_value()
        reverse_item = self.reverse_item.get_value()
        for c in range(self.num_clusters):
            end = beg + cluster_sizes[c]
            item_cluster[beg:end] = c
            item_pos_in_cluster[beg:end] = range(end - beg)
            reverse_item[c, 0:(end-beg)] = range(beg, end)
            beg = end
        self.item_cluster.set_value(item_cluster)
        self.item_pos_in_cluster.set_value(item_pos_in_cluster)
        self.reverse_item.set_value(reverse_item)

        self.do_kmeans(max_iters=1)         # do only one ; more will be done later

    @property
    def biggest_cluster_size(self):
        return self.W.shape[1]

    def cluster_mask(self, cluster_sizes):
        return tensor.lt(tensor.arange(self.biggest_cluster_size)[None, :],
                         cluster_sizes[:, None])

    def compile_kmeans_fun(self):
        # How To do k-means :
        # - Caluclate MIPS->MCSS version of W, b
        # - Recalculate centroids (doesn't even need a scan)
        # - Flatten W, b and corresponding mask
        # - Reaffect items by reconstructing each class (with a scan),
        #   using reverse_item as class description
        # - Loop
        # - Try to do the previous algorithm without ever using item_cluster and
        #   item_pos_in_cluster ; reconstruct them at the end manually with a for loop
        #   (see code for k_means)

        logger.info("Compiling k-means function...")

        # dimension of a transformed vector
        tdim = self.readout_dim + 1 + self.mips_to_mcss_params['m']

        # Apply MIPS->MCSS transform to W and b
        tvecs = mips_to_mcss_with_bias(
                   self.W.reshape((self.num_clusters * self.biggest_cluster_size,
                                   self.readout_dim)),
                   self.b.reshape((self.num_clusters * self.biggest_cluster_size,)),
                   **self.mips_to_mcss_params)
        tvecs = tvecs.reshape((self.num_clusters, self.biggest_cluster_size, tdim))

        # Generate a mask for all the classes (according to previous clustering)
        cluster_mask = self.cluster_mask(self.cluster_sizes)
                            
        # Calculate new centroids
        new_sums = (tvecs * cluster_mask[:, :, None]).sum(axis=1)
        new_norms = tensor.sqrt((new_sums ** 2).sum(axis=1, keepdims=True))
        new_centroids = new_sums / (new_norms + tensor.eq(new_norms, 0))

        """
        # If we have an empty cluster, replace its centroid with the centroid of the
        # biggest cluster plus some perturbation
        smallest = self.cluster_sizes.argmin()
        biggest = self.cluster_sizes.argmax()
        biggest_centroid = new_centroids[biggest, :]
        candidate_centroid = biggest_centroid + \
                self.theano_rng.normal(size=biggest_centroid.shape, std=0.001)
        new_centroids = ifelse(tensor.eq(self.cluster_sizes[smallest], 0),
                               tensor.set_subtensor(new_centroids[smallest], candidate_centroid),
                               new_centroids)
        """

        # Calculate new best cluster for the points, storing them in the
        # same fashion as the W and b are already stored (ie according to
        # the old clustering)
        new_bestclus = tensor.dot(tvecs, new_centroids.T).argmax(axis=2) * cluster_mask \
                        + (cluster_mask - 1)

        # Calculate number of items that change cluster (we stop when this is zero)
        num_changed = tensor.sum(tensor.neq(new_bestclus,
                                            tensor.arange(self.num_clusters)[:, None])
                                    * cluster_mask)

        # Flatten all the data, ie undo the clustering (some places are still unused,
        # there is a mask). This is simpler for when we do the eq-nonzero thing in the
        # scan later on
        new_bestclus_f = new_bestclus.reshape((self.num_clusters * self.biggest_cluster_size,))
        W_f = self.W.reshape((self.num_clusters * self.biggest_cluster_size, self.readout_dim))
        b_f = self.b.reshape((self.num_clusters * self.biggest_cluster_size,))
        reverse_item_f = self.reverse_item.reshape((self.num_clusters * self.biggest_cluster_size,))


        def build_cluster(i):
            # Find the indices (in the flattenned version) of all the items belonging
            # to cluster i according to the new clustering
            idxs = tensor.eq(new_bestclus_f, i).nonzero()[0]

            # Calculate how much padding must be added to the cluster
            npads = self.cluster_max_size - idxs.shape[0]

            # Return new cluster information: W, b, identity of selected item, and cluster size
            return [tensor.concatenate(
                            [mtx, tensor.zeros_like(mtx[0:1].repeat(axis=0, repeats=npads))],
                            axis=0)
                    for mtx in [W_f[idxs], b_f[idxs], reverse_item_f[idxs]]] + [idxs.shape[0]]
        [new_W, new_b, new_reverse_item, new_cluster_sizes], _ = \
                theano.map(build_cluster,
                           sequences=[tensor.arange(self.num_clusters)])

        num_empty = tensor.eq(new_cluster_sizes, 0).sum()

        # Trim new clustering data to save time & space
        new_max_clus_size = new_cluster_sizes.max()
        new_W = new_W[:, :new_max_clus_size, :]
        new_b = new_b[:, :new_max_clus_size]
        new_reverse_item = new_reverse_item[:, :new_max_clus_size]

        # Function that does one step of clustering and returns the number of items that 
        # have changed clusters
        self.kmeans_fun = theano.function(
                              inputs=[],
                              outputs=[num_changed, new_max_clus_size, num_empty],
                              updates=[
                                (self.W, new_W),
                                (self.b, new_b),
                                (self.reverse_item, new_reverse_item),
                                (self.centroids, new_centroids),
                                (self.cluster_sizes, new_cluster_sizes),
                              ])
        logger.info("Done compiling k-means function")

    def rebuild_item_idx(self):
        logger.info("Rebuilding item_culster,item_pos_in_cluster index...")

        cluster_sizes = self.cluster_sizes.get_value()
        reverse_item = self.reverse_item.get_value()
        
        item_cluster = -numpy.ones_like(self.item_cluster.get_value())
        item_pos_in_cluster = -numpy.ones_like(self.item_pos_in_cluster.get_value())

        for c in range(self.num_clusters):
            for i in range(cluster_sizes[c]):
                item = reverse_item[c, i]
                item_cluster[item] = c
                item_pos_in_cluster[item] = i

        self.item_cluster.set_value(item_cluster)
        self.item_pos_in_cluster.set_value(item_pos_in_cluster)

    def do_kmeans(self, max_iters=None):
        if self.kmeans_fun == None:
            self.compile_kmeans_fun()

        # Now we can do the actual k-means: loop until we're done
        it = 0
        while True:
            it = it + 1
            num_ch, new_max_clus_size, num_empty = self.kmeans_fun()
            logger.info("k-means iteration #{} : {} changed, biggest cluster is {}, {} empty clusters"
                    .format(it, num_ch, new_max_clus_size, num_empty))
            if num_ch == 0: break
            if max_iters is not None and it >= max_iters:
                break

        # Rebuild item_cluster and item_pos_in_cluster
        self.rebuild_item_idx()

    def generative_items_and_probs(self, readouts):
        batch_size = readouts.shape[0]

        # Do the MIPS->MCSS transform on the readout vectors
        trans_readouts = mips_to_mcss_with_bias_query(readouts, **self.mips_to_mcss_params)

        # Find the k best clusters for limiting our search
        clus_p = tensor.dot(trans_readouts, self.centroids.T)
        best_clus = clus_p.argsort(axis=1)[:, -self.emit_k_best_clusters:]

        final_shape = (batch_size,
                       best_clus.shape[1] * self.biggest_cluster_size)

        # Calculate a mask for the items in the cluster we selected
        mask = self.cluster_mask(self.cluster_sizes[best_clus.flatten()])
        mask = mask.reshape(final_shape)

        # This index maps items in cluster to their identity as a class number
        # in the whole input
        items = self.reverse_item[best_clus.flatten(), :]
        items = items.reshape(final_shape)

        # Calculate energies and softmax for the selected clusters
        energies = sparse_block_dot(self.W[None, :, :, :].dimshuffle(0, 1, 3, 2),
                                    readouts[:, None, :],
                                    tensor.zeros((batch_size, 1), dtype='int64'),
                                    self.b,
                                    best_clus)
        energies = energies.reshape(final_shape)
                            
        probs = weighted_softmax(energies, mask)

        return items, probs

    @application
    def emit(self, readouts):
        batch_size = readouts.shape[0]

        items, probs = self.generative_items_and_probs(readouts)

        gen = self.theano_rng.multinomial(pvals=probs).argmax(axis=1)

        return items[tensor.arange(batch_size), gen]

    @application
    def cost(self, readouts, outputs):
        # Reshape inputs so that readouts has two dimensions with fixed roles
        # (the first is whatever is used by the outside world, the second is the
        #  size of the readout)
        outputs_orig_shape = outputs.shape
        outputs = outputs.flatten()
        readouts = readouts.reshape((outputs.shape[0], readouts.shape[-1]))

        batch_size = readouts.shape[0]

        # Transform readouts MIPS->MCSS and find k best matching cluters
        # Also extract the sizes of those clusters as we need them later to
        # affect a weight to the elements of the randomly selected cluster
        trans_readouts = mips_to_mcss_with_bias_query(readouts, **self.mips_to_mcss_params)

        clus_p = tensor.dot(trans_readouts, self.centroids.T)
        best_clus = clus_p.argsort(axis=1)[:, -self.cost_k_best_clusters:]
        best_clus_nitems = self.cluster_sizes[best_clus.flatten()].reshape(best_clus.shape)

        # Find the target cluster and corresponding number of items
        target_clus = self.item_cluster[outputs][:, None]
        target_clus_nitems = self.cluster_sizes[target_clus.flatten()].reshape(target_clus.shape)

        # Sample a random cluster with the size of the cluster as probability
        # of selecting it
        clus_probs = tensor.cast(self.cluster_sizes, theano.config.floatX) / \
                     numpy.float32(self.output_dim)
        clus_probs_r = clus_probs[None, :].repeat(axis=0, repeats=batch_size)
        random_clus = self.theano_rng.multinomial(pvals=clus_probs_r).argmax(axis=1, keepdims=True)
        random_clus_nitems = self.cluster_sizes[random_clus.flatten()].reshape(random_clus.shape)

        # Affect a weight to random cluster to account for the fact that it represents
        # ALL the items that were not selected either as target or as best match
        random_clus_weights = tensor.cast(self.output_dim
                                            - best_clus_nitems.sum(axis=1, keepdims=True)
                                            - target_clus_nitems,
                                          dtype=theano.config.floatX) \
                              / tensor.cast(random_clus_nitems, dtype=theano.config.floatX)
        # FIRST RESULTS: our approach does not work. Experiment: set random cluster's weight
        # to one, like everybody else.
        # random_clus_weights = tensor.ones_like(random_clus_weights)

        # Concatenate all selected clusters into a single cluster choice matrix.
        # The target cluster is always first in this matrix so that extracting the
        # target element is always a straightforward process (see return at end of function)
        selected_clus = tensor.concatenate([target_clus, random_clus, best_clus], axis=1)
        selected_clus = theano.gradient.disconnected_grad(selected_clus)

        # selected_clus = theano.printing.Print("selected_clus")(selected_clus)

        flat_shape = (batch_size,
                      selected_clus.shape[1] * self.biggest_cluster_size)
        final_shape = (batch_size,
                       selected_clus.shape[1],
                       self.biggest_cluster_size)

        # Affect weights for all clusters (target and best have weight of 1)
        weights_by_cluster = tensor.concatenate([tensor.ones(target_clus.shape),
                                                 random_clus_weights,
                                                 tensor.ones(best_clus.shape)],
                                                axis=1)
        
        # Mask clusters that are selected several times (such as being both target
        # and best match), by setting the weight to zero for the extra occurences.
        # The first cluster (the target cluster) always keeps its weight of 1.
        def mask_duplicate_cluster(i):
            w = weights_by_cluster[:, i]
            if i == 0:
                return w[None, :]
            else:
                clus_eq = tensor.eq(selected_clus[:, i][:, None], selected_clus[:, :i])
                already_exists = clus_eq.any(axis=1)
                return tensor.switch(already_exists, w.zeros_like(), w)[None, :]
        weights_by_cluster = tensor.concatenate(map(mask_duplicate_cluster,
                                                    range(2 + self.cost_k_best_clusters)),
                                                axis=0)
        weights_by_cluster = weights_by_cluster.T
        weights_by_cluster = theano.gradient.disconnected_grad(weights_by_cluster)

        # weights_by_cluster = theano.printing.Print("weights_by_cluster")(weights_by_cluster)

        # Calculate a weight matrix for each individual element of the selected clusters
        # by combining the cluster-specific weights and a mask for the unused items
        # in cluster storage space
        weights = self.cluster_mask(self.cluster_sizes[selected_clus.flatten()])\
                        .reshape((selected_clus.shape[0],
                                  selected_clus.shape[1],
                                  self.biggest_cluster_size)) \
                  * weights_by_cluster[:, :, None]
        weights = weights.reshape(flat_shape)

        # weights = theano.printing.Print("weights")(weights)

        W = self.W
        # W = theano.printing.Print("W")(W)
        b = self.b
        # b = theano.printing.Print("b")(b)
        # readouts = theano.printing.Print('readouts')(readouts)

        # Calculates energies and corresponding softmax probabilities
        energies = sparse_block_dot(W[None, :, :, :].dimshuffle(0, 1, 3, 2),
                                    readouts[:, None, :],
                                    tensor.zeros((batch_size, 1), dtype='int64'),
                                    b,
                                    selected_clus)
        energies = energies.reshape(flat_shape)

        # energies = theano.printing.Print("energies")(energies)

        probs = weighted_softmax(energies, weights)
        probs = probs.reshape(final_shape)

        # probs = theano.printing.Print("probs")(probs)

        # Get probabilities of the targets
        target_prob = probs[tensor.arange(batch_size),
                            tensor.zeros((batch_size,), dtype='int64'),
                            self.item_pos_in_cluster[outputs]].reshape(outputs_orig_shape)

        # target_prob = theano.printing.Print("target_prob")(target_prob)

        # Return log likelihood
        return -tensor.log(target_prob)

    @application
    def initial_outputs(self, batch_size):
        return self.initial_output * tensor.ones((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        return super(ClusteredSoftmaxEmitter, self).get_dim(name)


class ClusteredSoftmaxDecoder(BaseDecoder):
    def __init__(self, clustering_args,**kwargs):
        emitter = ClusteredSoftmaxEmitter(
                    readout_dim=kwargs['embedding_dim'],
                    output_dim=kwargs['vocab_size'],
                    initial_output=-1,
                    **clustering_args)
        super(ClusteredSoftmaxDecoder, self).__init__(emitter=emitter,
                                                      **kwargs)


class ReclusterExtension(SimpleExtension):
    def __init__(self, emitter, max_iters, **kwargs):
        self.emitter = emitter
        self.max_iters = max_iters

        super(ReclusterExtension, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        self.emitter.do_kmeans(max_iters=self.max_iters)

