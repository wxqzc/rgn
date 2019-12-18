""" Geometric TensorFlow operations for protein structure prediction.

    There are some common conventions used throughout.

    BATCH_SIZE is the size of the batch, and may vary from iteration to iteration.
    NUM_STEPS is the length of the longest sequence in the data set (not batch). It is fixed as part of the tf graph.
    NUM_DIHEDRALS is the number of dihedral angles per residue (phi, psi, omega). It is always 3.
    NUM_DIMENSIONS is a constant of nature, the number of physical spatial dimensions. It is always 3.

    In general, this is an implicit ordering of tensor dimensions that is respected throughout. It is:

        NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS, NUM_DIMENSIONS

    The only exception is when NUM_DIHEDRALS are fused into NUM_STEPS. Btw what is setting the standard is the builtin 
    interface of tensorflow.models.rnn.rnn, which expects NUM_STEPS x [BATCH_SIZE, NUM_AAS].
"""

__author__ = "Mohammed AlQuraishi and Hermes Spaink"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

# Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import collections
from iminuit import Minuit

# Constants
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = np.array([145.801, 152.326, 132.868], dtype='float32')
BOND_ANGLES  = np.array([  2.124,   1.941,   2.028], dtype='float32')

# Functions
def angularize(input_tensor, name=None):
    """ Restricts real-valued tensors to the interval [-pi, pi] by feeding them through a cosine. """

    with tf.name_scope(name, 'angularize', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
    
        return tf.multiply(np.pi, tf.cos(input_tensor + (np.pi / 2)), name=scope)

def reduce_mean_angle(weights, angles, use_complex=False, name=None):
    """ Computes the weighted mean of angles. Accepts option to compute use complex exponentials or real numbers.

        Complex number-based version is giving wrong gradients for some reason, but forward calculation is fine.

        See https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    Args:
        weights: [BATCH_SIZE, NUM_ANGLES]
        angles:  [NUM_ANGLES, NUM_DIHEDRALS]

    Returns:
                 [BATCH_SIZE, NUM_DIHEDRALS]

    """

    with tf.name_scope(name, 'reduce_mean_angle', [weights, angles]) as scope:
        weights = tf.convert_to_tensor(weights, name='weights')
        angles  = tf.convert_to_tensor(angles,  name='angles')

        if use_complex:
            # use complexed-valued exponentials for calculation
            cwts =        tf.complex(weights, 0.) # cast to complex numbers
            exps = tf.exp(tf.complex(0., angles)) # convert to point on complex plane

            unit_coords = tf.matmul(cwts, exps) # take the weighted mixture of the unit circle coordinates

            return tf.angle(unit_coords, name=scope) # return angle of averaged coordinate

        else:
            # use real-numbered pairs of values
            sins = tf.sin(angles)
            coss = tf.cos(angles)

            y_coords = tf.matmul(weights, sins)
            x_coords = tf.matmul(weights, coss)

            return tf.atan2(y_coords, x_coords, name=scope)

def reduce_l2_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, epsilon=1e-12, name=None):
    """ Computes the (possibly weighted) L2 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """

    with tf.name_scope(name, 'reduce_l2_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
        
        input_tensor_sq = tf.square(input_tensor)
        if weights is not None: input_tensor_sq = input_tensor_sq * weights

        return tf.sqrt(tf.maximum(tf.reduce_sum(input_tensor_sq, axis=reduction_indices, keep_dims=keep_dims), epsilon), name=scope)

def reduce_l1_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, nonnegative=True, name=None):
    """ Computes the (possibly weighted) L1 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """

    with tf.name_scope(name, 'reduce_l1_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
        
        if not nonnegative: input_tensor = tf.abs(input_tensor)
        if weights is not None: input_tensor = input_tensor * weights

        return tf.reduce_sum(input_tensor, axis=reduction_indices, keep_dims=keep_dims, name=scope)

def dihedral_to_point(dihedral, r=BOND_LENGTHS, theta=BOND_ANGLES, name=None):
    """ Takes triplets of dihedral angles (omega, phi, psi) and returns 3D points ready for use in
        reconstruction of coordinates. Bond lengths and angles are based on idealized averages.

    Args:
        dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

    Returns:
                  [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    with tf.name_scope(name, 'dihedral_to_point', [dihedral]) as scope:
        dihedral = tf.convert_to_tensor(dihedral, name='dihedral') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

        num_steps  = tf.shape(dihedral)[0]
        batch_size = dihedral.get_shape().as_list()[1] # important to use get_shape() to keep batch_size fixed for performance reasons

        r_cos_theta = tf.constant(r * np.cos(np.pi - theta), name='r_cos_theta') # [NUM_DIHEDRALS]
        r_sin_theta = tf.constant(r * np.sin(np.pi - theta), name='r_sin_theta') # [NUM_DIHEDRALS]

        pt_x = tf.tile(tf.reshape(r_cos_theta, [1, 1, -1]), [num_steps, batch_size, 1], name='pt_x') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_y = tf.multiply(tf.cos(dihedral), r_sin_theta,                               name='pt_y') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_z = tf.multiply(tf.sin(dihedral), r_sin_theta,                               name='pt_z') # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]

        pt = tf.stack([pt_x, pt_y, pt_z])                                                       # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
        pt_perm = tf.transpose(pt, perm=[1, 3, 2, 0])                                           # [NUM_STEPS, NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
        pt_final = tf.reshape(pt_perm, [num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS], # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMS]
                              name=scope) 

        return pt_final

def point_to_coordinate(pt, num_fragments=6, parallel_iterations=4, swap_memory=False, name=None):
    """ Takes points from dihedral_to_point and sequentially converts them into the coordinates of a 3D structure.

        Reconstruction is done in parallel, by independently reconstructing num_fragments fragments and then 
        reconstituting the chain at the end in reverse order. The core reconstruction algorithm is NeRF, based on 
        DOI: 10.1002/jcc.20237 by Parsons et al. 2005. The parallelized pNERF version is described in 
        DOI: 10.1002/jcc.25772 by AlQuraishi 2019.

    Args:
        pt: [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

    Opts:
        num_fragments: Number of fragments to reconstruct in parallel. If None, the number is chosen adaptively

    Returns:
            [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] 
    """                             

    with tf.name_scope(name, 'point_to_coordinate', [pt]) as scope:
        pt = tf.convert_to_tensor(pt, name='pt')

        # compute optimal number of fragments if needed
        s = tf.shape(pt)[0] # NUM_STEPS x NUM_DIHEDRALS
        if num_fragments is None: num_fragments = tf.cast(tf.sqrt(tf.cast(s, dtype=tf.float32)), dtype=tf.int32)

        # initial three coordinates (specifically chosen to eliminate need for extraneous matmul)
        Triplet = collections.namedtuple('Triplet', 'a, b, c')
        batch_size = pt.get_shape().as_list()[1] # BATCH_SIZE
        init_mat = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0], [-np.sqrt(2.0), 0, 0], [0, 0, 0]], dtype='float32')
        init_coords = Triplet(*[tf.reshape(tf.tile(row[np.newaxis], tf.stack([num_fragments * batch_size, 1])), 
                                           [num_fragments, batch_size, NUM_DIMENSIONS]) for row in init_mat])
                      # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        # pad points to yield equal-sized fragments
        r = ((num_fragments - (s % num_fragments)) % num_fragments)          # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
        pt = tf.pad(pt, [[0, r], [0, 0], [0, 0]])                            # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.reshape(pt, [num_fragments, -1, batch_size, NUM_DIMENSIONS]) # [NUM_FRAGS, FRAG_SIZE,  BATCH_SIZE, NUM_DIMENSIONS]
        pt = tf.transpose(pt, perm=[1, 0, 2, 3])                             # [FRAG_SIZE, NUM_FRAGS,  BATCH_SIZE, NUM_DIMENSIONS]

        # extension function used for single atom reconstruction and whole fragment alignment
        def extend(tri, pt, multi_m):
            """
            Args:
                tri: NUM_DIHEDRALS x [NUM_FRAGS/0,         BATCH_SIZE, NUM_DIMENSIONS]
                pt:                  [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
                multi_m: bool indicating whether m (and tri) is higher rank. pt is always higher rank; what changes is what the first rank is.
            """

            bc = tf.nn.l2_normalize(tri.c - tri.b, -1, name='bc')                                        # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]        
            n = tf.nn.l2_normalize(tf.cross(tri.b - tri.a, bc), -1, name='n')                            # [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMS]
            if multi_m: # multiple fragments, one atom at a time. 
                m = tf.transpose(tf.stack([bc, tf.cross(n, bc), n]), perm=[1, 2, 3, 0], name='m')        # [NUM_FRAGS,   BATCH_SIZE, NUM_DIMS, 3 TRANS]
            else: # single fragment, reconstructed entirely at once.
                s = tf.pad(tf.shape(pt), [[0, 1]], constant_values=3)                                    # FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS
                m = tf.transpose(tf.stack([bc, tf.cross(n, bc), n]), perm=[1, 2, 0])                     # [BATCH_SIZE, NUM_DIMS, 3 TRANS]
                m = tf.reshape(tf.tile(m, [s[0], 1, 1]), s, name='m')                                    # [FRAG_SIZE, BATCH_SIZE, NUM_DIMS, 3 TRANS]
            coord = tf.add(tf.squeeze(tf.matmul(m, tf.expand_dims(pt, 3)), axis=3), tri.c, name='coord') # [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMS]
            return coord
        
        # loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially generating the coordinates for each fragment across all batches
        i = tf.constant(0)
        s_padded = tf.shape(pt)[0] # FRAG_SIZE
        coords_ta = tf.TensorArray(tf.float32, size=s_padded, tensor_array_name='coordinates_array')
                    # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        def loop_extend(i, tri, coords_ta): # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
            coord = extend(tri, pt[i], True)
            return [i + 1, Triplet(tri.b, tri.c, coord), coords_ta.write(i, coord)]

        _, tris, coords_pretrans_ta = tf.while_loop(lambda i, _1, _2: i < s_padded, loop_extend, [i, init_coords, coords_ta],
                                                    parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                                      # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS], 
                                      # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS] 
        
        # loop over NUM_FRAGS in reverse order, bringing all the downstream fragments in alignment with current fragment
        coords_pretrans = tf.transpose(coords_pretrans_ta.stack(), perm=[1, 0, 2, 3]) # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        i = tf.shape(coords_pretrans)[0] # NUM_FRAGS

        def loop_trans(i, coords):
            transformed_coords = extend(Triplet(*[di[i] for di in tris]), coords, False)
            return [i - 1, tf.concat([coords_pretrans[i], transformed_coords], 0)]

        _, coords_trans = tf.while_loop(lambda i, _: i > -1, loop_trans, [i - 2, coords_pretrans[-1]],
                                        parallel_iterations=parallel_iterations, swap_memory=swap_memory)
                          # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]

        # lose last atom and pad from the front to gain an atom ([0,0,0], consistent with init_mat), to maintain correct atom ordering
        coords = tf.pad(coords_trans[:s-1], [[1, 0], [0, 0], [0, 0]], name=scope) # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]

        return coords

def drmsd(u, v, weights, name=None):
    """ Computes the dRMSD of two tensors of vectors.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u, v:    [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        weights: [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    Returns:
                 [BATCH_SIZE]
    """

    with tf.name_scope(name, 'dRMSD', [u, v, weights]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        v = tf.convert_to_tensor(v, name='v')
        weights = tf.convert_to_tensor(weights, name='weights')

        diffs = pairwise_distance(u) - pairwise_distance(v)                                  # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]
        norms = reduce_l2_norm(diffs, reduction_indices=[0, 1], weights=weights, name=scope) # [BATCH_SIZE]

        return norms

def pairwise_distance(u, name=None):
    """ Computes the pairwise distance (l2 norm) between all vectors in the tensor.

        Vectors are assumed to be in the third dimension. Op is done element-wise over batch.

    Args:
        u: [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]

    Returns:
           [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

    """
    with tf.name_scope(name, 'pairwise_distance', [u]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        
        diffs = u - tf.expand_dims(u, 1)                                 # [NUM_STEPS, NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        norms = reduce_l2_norm(diffs, reduction_indices=[3], name=scope) # [NUM_STEPS, NUM_STEPS, BATCH_SIZE]

        return norms


def tmscore(u, v, weights, num_steps, batch_size, name=None):
    """ Computes the tm_score of two tensors of vectors.

        vectors are assumed to be in the third dimension. Op is done element-wise over batch.
    Args:
        u, v:      [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS]
        weights:   [NUM_STEPS, NUM_STEPS, BATCH_SIZE]
        num_steps: [BATCH_SIZE]
        batch_size: constant

    Returns:
                   [BATCH_SIZE]
    """

    with tf.name_scope(name, 'TM_score', [u, v, weights, num_steps]) as scope:
        u = tf.convert_to_tensor(u, name='u')
        v = tf.convert_to_tensor(v, name='v')
        weights = tf.convert_to_tensor(weights, name='weights')
        num_steps = tf.convert_to_tensor(num_steps, name='num_steps')

        # Prepare data and concat into one tensor along NUM_STEPS dimension
        c1 = tf.map_fn(lambda x: tf.concat((x, tf.ones([batch_size, 1], dtype=tf.float32)), axis=-1),
                       u)  # [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS+1]
        c2 = tf.map_fn(lambda x: tf.concat((x, tf.ones([batch_size, 1], dtype=tf.float32)), axis=-1),
                       v)  # [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS+1]
        coords = tf.concat([c1, c2], axis=0)              # [2*NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS+1]
        coords = tf.transpose(coords, perm=[1, 0, 2])     # [BATCH_SIZE, 2*NUM_STEPS, NUM_DIMENSIONS+1]

        # d02 = (1.24 * (num_steps - 15.) ** (1. / 3.) - 1.8) ** 2.
        d02 = (tf.constant(1.24) * (tf.dtypes.cast(num_steps, tf.float32) - tf.constant(15.)) ** tf.constant(
            (1. / 3.)) - tf.constant(1.8)) ** tf.constant(2.)                             # [BATCH_SIZE]
        dx, dy, dz, theta, phi, psi = get_default_values(c1, c2, batch_size, name=scope)  # [BATCH_SIZE]
        params = tf.stack([dx, dy, dz, theta, phi, psi, d02], axis=1)                     # [BATCH_SIZE, 7]
        params = tf.reshape(params, (batch_size, 7, 1))                                   # [BATCH_SIZE, 7, 1]
        params = tf.pad(params, [[0, 0], [0, 0], [0, 3]])                                 # [BATCH_SIZE, 7, 4]

        data = tf.concat([coords, params], axis=1)  # [BATCH_SIZE, 2*NUM_STEPS+7, NUM_DIMENSIONS+1]

        tm_scores = tf.scan(minimize_tm, data, name=scope)  # [BATCH_SIZE]  # TODO fix dit! Zie docs

        return tm_scores


def minimize_tm(_, data, name=None):
    """ Performs the tm_score minimization on a single coordinate pair.

    Args:
         data: [2*NUM_STEPS+7, NUM_DIMENSIONS+1]

    Returns:
                 []
    """

    with tf.name_scope(name, 'minimized_tm_score', [data]) as scope:
        data = tf.convert_to_tensor(data, name='coords')

        # TODO Gaat dit werken??
        # Unpack data
        coords = data[:-7, :]  # [2*NUM_STEPS, NUM_DIMENSIONS+1]
        params = data[-7:, 0]  # [7]
        c1, c2 = tf.split(coords, 2, axis=0)  # [NUM_STEPS, NUM_DIMENSIONS+1]
        dx, dy, dz, theta, phi, psi, d02 = [tf.squeeze(i) for i in tf.split(params, 7, axis=0)]  # []

        # Define scoring methods
        def tm_score(dx, dy, dz, theta, phi, psi):
            matrix = get_matrix(dx, dy, dz, theta, phi, psi, name=scope)  # [NUM_DIMENSIONS+1, NUM_DIMENSIONS+1]
            dist = tf.matmul(c2, matrix) - c1                             # [NUM_STEPS, NUM_DIMENSIONS+1]
            d_i2 = tf.reduce_sum(dist, axis=-1) ** 2                      # [NUM_STEPS]

            one = tf.constant(1.)
            tm = one / (one + (d_i2 / d02))                               # [NUM_STEPS]
            return tm

        def tm_sum(dx, dy, dz, theta, phi, psi):
            return tf.reduce_sum(tm_score(dx, dy, dz, theta, phi, psi))

        # Minimize parameters

        # m = Minuit(tm_sum,
        #            error_dx=1., error_dy=1., error_dz=1.,
        #            error_theta=.01, error_phi=.01, error_psi=.01,
        #            dx=dx, dy=dy, dz=dz,
        #            theta=theta, phi=phi, psi=psi,
        #            pedantic=False, print_level=0,
        #            )
        # m.migrad()

        initial_position = (dx, dy, dz, theta, phi, psi)
        # print(initial_position)
        res = tfp.optimizer.differential_evolution_minimize(
            tm_sum,
            initial_position=initial_position,
            # population_stddev=2.,
            seed=42,
        )
        dx, dy, dz, theta, phi, psi = res[2]

        return tf.reduce_mean(tm_score(dx, dy, dz, theta, phi, psi))  # []


def get_matrix(dx, dy, dz, theta, phi, psi, name=None):
    """ Compute rotation-translation matrix from angles and displacements

    Args:
         dx, dy, dz, theta, phi, psi: []

    Returns:
        [4, 4]
    """

    with tf.name_scope(name, 'get_matrix', [dx, dy, dz, theta, phi, psi]) as scope:
        dx = tf.convert_to_tensor(dx, name='dx')
        dy = tf.convert_to_tensor(dy, name='dy')
        dz = tf.convert_to_tensor(dz, name='dz')
        theta = tf.convert_to_tensor(theta, name='theta')
        phi = tf.convert_to_tensor(phi, name='phi')
        psi = tf.convert_to_tensor(psi, name='psi')

        cx = tf.cos(theta)
        cy = tf.cos(phi)
        cz = tf.cos(psi)

        sx = tf.sin(theta)
        sy = tf.sin(phi)
        sz = tf.sin(psi)

        nul = tf.constant(0.)
        matrix = tf.stack([[ cx * cz - sx * cy * sz,  cx * sz + sx * cy * cz, sx * sy, dx],
                           [-sx * cz - cx * cy * sz, -sx * sz + cx * cy * cz, cx * sy, dy],
                           [                sy * sz,                -sy * cz,      cy, dz],
                           [nul, nul, nul, tf.constant(1.)]])
        return matrix


def get_default_values(c1, c2, batch_size, name=None):
    """ Make a crude estimation of the alignment using the center of mass
        and general orientation of the protein.

    Args:
        c1, c2: [NUM_STEPS, BATCH_SIZE, NUM_DIMENSIONS+1]

    Returns:
        dx, dy, dz, theta, phi, psi:      [BATCH_SIZE]

    """

    with tf.name_scope(name, 'default_vals', [c1, c2]) as scope:
        c1 = tf.convert_to_tensor(c1, name='c1')
        c2 = tf.convert_to_tensor(c2, name='c2')

        dx, dy, dz, _ = tf.unstack(tf.reduce_mean(c1 - c2, axis=0), axis=1)     # [BATCH_SIZE]

        vec1 = tf.reduce_mean(tf.split(c1, [3, 1], axis=-1)[0], axis=0)         # [BATCH_SIZE, NUM_DIMENSIONS]
        vec2 = tf.reduce_mean(tf.split(c2, [3, 1], axis=-1)[0], axis=0)         # [BATCH_SIZE, NUM_DIMENSIONS]

        #  Find the rotation matrix that converts vec1 into vec2
        #  http://math.stackexchange.com/questions/180418/#476311
        v = tf.linalg.cross(vec1, vec2)                                         # [BATCH_SIZE, NUM_DIMENSIONS]
        s = tf.add(tf.norm(v, axis=-1), tf.constant(np.finfo(np.float32).eps))  # [BATCH_SIZE]
        c = tf.reduce_sum(tf.multiply(vec1, vec2), axis=-1)                     # [BATCH_SIZE]

        #  skew-symmetric cross-product matrix
        def sscpm(v):
            v0, v1, v2 = tf.split(v, 3, axis=-1)
            v0, v1, v2 = tf.squeeze(v0), tf.squeeze(v1), tf.squeeze(v2)
            nul = tf.constant(.0)
            return tf.stack([tf.stack([nul, -v2, v1]),
                             tf.stack([v2, nul, -v0]),
                             tf.stack([-v1, v0, nul])])

        vx = tf.map_fn(sscpm, v)  # [BATCH_SIZE, NUM_DIMENSIONS, NUM_DIMENSIONS]
        rot_mat = tf.eye(3, batch_shape=[batch_size]) + vx + (vx ** tf.constant(2.)) * tf.reshape(
                                                                (tf.constant(1.) - c) / (s ** tf.constant(2.)),
                                                                (batch_size, 1, 1))

        # Recover the angles from the matrix as seen here:
        # http://nghiaho.com/?page_id=846
        def _theta(rm):
            return tf.atan2(rm[2, 1], rm[2, 2])

        def _phi(rm):
            return tf.atan2(-rm[2, 0], tf.sqrt(rm[2, 1] ** tf.constant(2.) + rm[2, 2] ** tf.constant(2.)))

        def _psi(rm):
            return tf.atan2(rm[1, 0], rm[0, 0])

        theta = tf.map_fn(_theta, rot_mat)
        phi   = tf.map_fn(_phi, rot_mat)
        psi   = tf.map_fn(_psi, rot_mat)

        return dx, dy, dz, theta, phi, psi
