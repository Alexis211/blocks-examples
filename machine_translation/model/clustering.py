import logging
import theano
import numpy
from theano import tensor

from theano.sandbox.blocksparse import sparse_block_dot

from blocks.roles import add_role, WEIGHT, BIAS

from blocks.bricks.sequence_generators import AbstractEmitter
from blocks.bricks.base import application
from blocks.bricks import Linear, Bias, Initializable, Random

from blocks.utils import shared_floatx_nans

from decoder import BaseDecoder

from model import NO_GRADIENT


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
    emax = energies.max(axis=1, keepdims=True)
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
        self.cluster_sizes = theano.shared(numpy.zeros(self.num_clusters).astype(numpy.int32),
                                         name='cluster_sizes')
        # Lookup table for each item: in which class and at which position is it?
        self.item_class = theano.shared(-numpy.ones(self.output_dim).astype(numpy.int32),
                                        name='item_class')
        self.item_pos_in_class = theano.shared(-numpy.ones(self.output_dim).astype(numpy.int32),
                                               name='item_pos_in_class')
        # Reverse lookup table: for each item in a class, what is the actual item
        self.reverse_item = theano.shared(-numpy.ones((self.num_clusters, self.cluster_max_size))
                                                .astype(numpy.int32),
                                          name='reverse_item')

        add_role(self.W, WEIGHT)
        add_role(self.b, BIAS)
        add_role(self.centroids, NO_GRADIENT)
        add_role(self.cluster_sizes, NO_GRADIENT)
        add_role(self.item_class, NO_GRADIENT)
        add_role(self.item_pos_in_class, NO_GRADIENT)
        add_role(self.reverse_item, NO_GRADIENT)
        self.parameters = [self.W, self.b,
                           self.item_class, self.item_pos_in_class,
                           self.reverse_item]

    def _initialize(self):
        # How To Initialize:
        # - Set cluster_sizes so that vectors are equally divided into classes
        # - Do something (manually with a for loop, it's okay!) so that item_class,
        #   item_pos_in_class and reverse_item have consistent values (eg. sequentially
        #   attribute items to classes and W,b item pairs)
        # - call do_kmeans

        self.kmeans_fun = None

        # Initialize W and b to random values
        self.weights_init.initialize(self.W, self.rng)
        self.biases_init.initialize(self.b, self.rng)

        # Calculate class sizes
        cluster_sizes = (self.output_dim / self.num_clusters) * \
                      numpy.ones(self.num_clusters, dtype='int32')
        n_bigger_classes = self.output_dim % self.num_clusters
        cluster_sizes[:n_bigger_classes] += 1
        self.cluster_sizes.set_value(cluster_sizes)

        # Fill classes
        i = 0
        item_class = self.item_class.get_value()
        item_pos_in_class = self.item_pos_in_class.get_value()
        reverse_item = self.reverse_item.get_value()
        for c in range(self.num_clusters):
            beg = i
            end = i + cluster_sizes[c]
            item_class[beg:end] = c
            item_pos_in_class[beg:end] = range(end - beg)
            reverse_item[c, 0:(end-beg)] = range(beg, end)
        self.item_class.set_value(item_class)
        self.item_pos_in_class.set_value(item_pos_in_class)
        self.reverse_item.set_value(reverse_item)

        self.do_kmeans()

    def class_mask(self, cluster_sizes):
        return tensor.lt(tensor.arange(self.cluster_max_size, dtype='int32')[None, :],
                         cluster_sizes[:, None])

    def do_kmeans(self):
        # How To do k-means :
        # - Caluclate MIPS->MCSS version of W, b
        # - Recalculate centroids (doesn't even need a scan)
        # - Flatten W, b and corresponding mask
        # - Reaffect items by reconstructing each class (with a scan),
        #   using reverse_item as class description
        # - Loop
        # - Try to do the previous algorithm without ever using item_class and
        #   item_pos_in_class ; reconstruct them at the end manually with a for loop
        
        # dimension of a transformed vector
        tdim = self.readout_dim + 1 + self.mips_to_mcss_params['m']

        if self.kmeans_fun == None:
            logger.info("Compiling k-means function...")

            # Apply MIPS->MCSS transform to W and b
            tvecs = mips_to_mcss_with_bias(
                       self.W.reshape((self.num_clusters * self.cluster_max_size,
                                       self.readout_dim)),
                       self.b.reshape((self.num_clusters * self.cluster_max_size,)),
                       **self.mips_to_mcss_params)
            tvecs = tvecs.reshape((self.num_clusters, self.cluster_max_size, tdim))

            # Generate a mask for all the classes (according to previous clustering)
            class_mask = self.class_mask(self.cluster_sizes)
                                
            # Calculate new centroids
            new_sums = (tvecs * class_mask[:, :, None]).sum(axis=1)
            new_norms = tensor.sqrt((new_sums ** 2).sum(axis=1, keepdims=True))
            new_centroids = new_sums / (new_norms + tensor.eq(new_norms, 0))

            # Calculate new best cluster for the points, storing them in the
            # same fashion as the W and b are already stored (ie according to
            # the old clustering)
            new_bestclus = tensor.dot(tvecs, new_centroids.T).argmax(axis=2) * class_mask \
                            + (class_mask - 1)

            # Calculate number of items that change cluster (we stop when this is zero)
            num_changed = tensor.sum(tensor.neq(new_bestclus,
                                                tensor.arange(self.num_clusters)[:, None])
                                        * class_mask)

            # Flatten all the data, ie undo the clustering (some places are still unused,
            # there is a mask). This is simpler for when we do the eq-nonzero thing in the
            # scan later on
            new_bestclus_f = new_bestclus.reshape((self.num_clusters * self.cluster_max_size,))
            W_f = self.W.reshape((self.num_clusters * self.cluster_max_size, self.readout_dim))
            b_f = self.b.reshape((self.num_clusters * self.cluster_max_size,))
            reverse_item_f = self.reverse_item.reshape((self.num_clusters * self.cluster_max_size,))


            def build_cluster(i):
                # Find the indices (in the flattenned version) of all the items belonging
                # to cluster i according to the new clustering
                idxs = tensor.eq(new_bestclus_f, i).nonzero()[0]

                # Calculate how much padding must be added to the cluster
                npads = self.cluster_max_size - idxs.shape[0]

                # Return new cluster information: W, b, identity of selected item, and cluster size
                return [tensor.concatenate([mtx, mtx[0:1].repeat(axis=0, repeats=npads)], axis=0)
                        for mtx in [W_f[idxs], b_f[idxs], reverse_item_f[idxs]]] + [idxs.shape[0]]
            [new_W, new_b, new_reverse_item, new_cluster_sizes], _ = \
                    theano.map(build_cluster,
                               sequences=[tensor.arange(self.num_clusters)])

            # Function that does one step of clustering and returns the number of items that 
            # have changed clusters
            self.kmeans_fun = theano.function(
                                  inputs=[],
                                  outputs=[num_changed],
                                  updates=[
                                    (self.W, new_W),
                                    (self.b, new_b),
                                    (self.reverse_item, new_reverse_item),
                                    (self.centroids, new_centroids),
                                    (self.cluster_sizes, tensor.cast(new_cluster_sizes, 'int32')),
                                  ])
            logger.info("Done compiling k-means function")

        # Now we can do the actual k-means: loop until we're done
        it = 0
        while True:
            it = it + 1
            num_ch, = self.kmeans_fun()
            logger.info("k-means iteration #{} : {} changed".format(it, num_ch))
            if num_ch == 0: break
            if it > 5: break    # TODO this is for debugging purposes only (makes stuff faster)

        # Rebuild item_class and item_pos_in_class
        item_class = self.item_class.get_value()
        item_pos_in_class = self.item_pos_in_class.get_value()
        cluster_sizes = self.cluster_sizes.get_value()
        reverse_item = self.reverse_item.get_value()
        for c in range(self.num_clusters):
            for i in range(cluster_sizes[c]):
                item = reverse_item[c, i]
                item_class[item] = c
                item_pos_in_class[item] = i
        self.item_class.set_value(item_class)
        self.item_pos_in_class.set_value(item_pos_in_class)

    @application
    def emit(self, readouts):
        batch_size = readouts.shape[0]

        trans_readouts = mips_to_mcss_with_bias_query(readouts, **self.mips_to_mcss_params)

        clus_p = tensor.dot(trans_readouts, self.centroids.T)
        best_clus = clus_p.argsort(axis=1)[:, -self.emit_k_best_clusters:]

        final_shape = (batch_size,
                       best_clus.shape[1] * self.cluster_max_size)

        mask = self.class_mask(self.cluster_sizes[best_clus.flatten()])
        mask = mask.reshape(final_shape)

        items = self.reverse_item[best_clus.flatten(), :]
        items = items.reshape(final_shape)

        energies = sparse_block_dot(self.W[None, :, :, :],
                                    readouts[:, None, :],
                                    tensor.zeros((batch_size, 1), dtype='int32'),
                                    self.b,
                                    best_clus)
        energies = energies.reshape(final_shape)
                            
        probs = weighted_softmax(energies, mask)

        gen = self.theano_rng.multinomial(pvals=probs).argmax(axis=1)
        return items[tensor.arange(batch_size), gen]

    @application
    def cost(self, readouts, outputs):
        outputs = outputs.flatten()
        readouts = readouts.reshape((outputs.shape[0], readouts.shape[-1]))

        batch_size = readouts.shape[0]

        trans_readouts = mips_to_mcss_with_bias_query(readouts, **self.mips_to_mcss_params)

        clus_p = tensor.dot(trans_readouts, self.centroids.T)
        best_clus = clus_p.argsort(axis=1)[:, -self.cost_k_best_clusters:]
        best_clus_nitems = self.cluster_sizes[best_clus.flatten()].reshape(best_clus.shape)

        target_clus = self.item_class[outputs.flatten()][:, None]
        target_clus_nitems = self.cluster_sizes[target_clus.flatten()].reshape(target_clus.shape)

        clus_probs = tensor.cast(self.cluster_sizes, theano.config.floatX) / \
                     numpy.float32(self.output_dim)
        clus_probs_r = clus_probs[None, :].repeat(axis=0, repeats=batch_size)
        random_clus = self.theano_rng.multinomial(pvals=clus_probs_r).argmax(axis=1, keepdims=True)
        random_clus_nitems = self.cluster_sizes[random_clus.flatten()].reshape(random_clus.shape)

        random_clus_weights = (self.output_dim - best_clus_nitems.sum(axis=1, keepdims=True)
                                               - target_clus_nitems
                                               - random_clus_nitems) / random_clus_nitems

        selected_clus = tensor.concatenate([target_clus, random_clus, best_clus], axis=1)
        selected_clus = theano.gradient.disconnected_grad(selected_clus)

        flat_shape = (batch_size,
                      selected_clus.shape[1] * self.cluster_max_size)
        final_shape = (batch_size,
                       selected_clus.shape[1],
                       self.cluster_max_size)

        weights_by_cluster = tensor.concatenate([tensor.ones(target_clus.shape),
                                                 random_clus_weights,
                                                 tensor.ones(best_clus.shape)],
                                                axis=1)

        def mask_duplicate_cluster(i):
            clus_eq = tensor.eq(selected_clus[:, i][:, None], selected_clus[:, :i])
            already_exists = clus_eq.max(axis=1, keepdims=True)
            w = weights_by_cluster[:, i][:, None]
            return tensor.switch(already_exists, w.zeros_like(), w)
        weights_by_cluster, _ = theano.map(mask_duplicate_cluster,
                                           sequences=tensor.arange(weights_by_cluster.shape[1]))

        weights = self.class_mask(self.cluster_sizes[selected_clus.flatten()]) * \
                  weights_by_cluster[:, :, None]
        weights = weights.reshape(flat_shape)

        energies = sparse_block_dot(self.W[None, :, :, :],
                                    readouts[:, None, :],
                                    tensor.zeros((batch_size, 1), dtype='int32'),
                                    self.b,
                                    selected_clus)
        energies = energies.reshape(flat_shape)

        probs = weighted_softmax(energies, weights)
        probs = probs.reshape(final_shape)

        return probs[tensor.arange(batch_size),
                     tensor.zeros((batch_size,), dtype='int32'),
                     self.item_pos_in_class[outputs]].reshape(outputs.shape)

    @application
    def initial_outputs(self, batch_size):
        return self.initial_output * tensor.ones((batch_size,), dtype='int32')

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



