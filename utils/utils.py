import numpy as np
import csv
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import time
import tensorflow as tf
import sklearn
from pathlib import Path
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard
from keras.optimizers import SGD
from keras.optimizers import Adam
from .threading import StoppableThread
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
from IPython.display import Image
from keras.optimizers import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_auc_score
from PIL import Image
import bisect
import model_lib
from joblib import Parallel, delayed
from skimage.transform import resize
from keras.utils import Progbar
from icecream import ic
from utils import graphs

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#K.tensorflow_backend.set_session(tf.Session(config=config))

def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
    """
    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:
    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array
    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
        .. versionadded:: 1.9.0
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.
        .. versionadded:: 1.13.0
    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.
        .. versionadded:: 1.9.0
    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    repeat : Repeat elements of an array.
    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    (move the axis to the first dimension to keep the order of the other axes)
    and then flattening the subarrays in C order. The flattened subarrays are
    then viewed as a structured type with each element given a label, with the
    effect that we end up with a 1-D array of structured types that can be
    treated in the same way as any other 1-D array. The result is that the
    flattened subarrays are sorted in lexicographic order starting with the
    first element.
    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])
    Return the unique rows of a 2D array
    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])
    Return the indices of the original array that give the unique values:
    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'], dtype='<U1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'], dtype='<U1')
    Reconstruct the input array from the unique values and inverse:
    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])
    Reconstruct the input values from the unique values and counts:
    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = np.unique(a, return_counts=True)
    >>> values
    array([1, 2, 3, 4, 6])
    >>> counts
    array([1, 3, 1, 1, 1])
    >>> np.repeat(values, counts)
    array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved
    """
    ar = np.asanyarray(ar)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts)
        return _unpack_tuple(ret)

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.AxisError(axis, ar.ndim) from None

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
        if ar.shape[1] > 0:
            consolidated = ar.view(dtype)
        else:
            # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
            # a data type with itemsize 0, and the call `ar.view(dtype)` will
            # fail.  Instead, we'll use `np.empty` to explicitly create the
            # array with shape `(len(ar),)`.  Because `dtype` in this case has
            # itemsize 0, the total size of the result is still 0 bytes.
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype)) from e

    def reshape_uniq(uniq):
        n = len(uniq)
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(n, *orig_shape[1:])
        uniq = np.moveaxis(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)

def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret

def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x

def print_images_clecio_like(x_data, num_prints, num_channels, input_shape, index):
    Path('/home/kayque/LENSLOAD/').parent
    os.chdir('/home/kayque/LENSLOAD/')

    source = '/home/kayque/LENSLOAD/'
    print(' ** Does an old folder exists?')
    if os.path.exists(source+'lens_yes_2'):
        print(' ** Yes, it does! Trying to delete... ')
        shutil.rmtree(source+'lens_yes_2', ignore_errors=True)
        print(" ** Supposedly done. Checking if there's an RNCV folder...")
        os.mkdir('lens_yes_2')
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir('lens_yes_2')
        print(' ** Done!')

    dest1 = ('/home/kayque/LENSLOAD/lens_yes_2/')
    counter = 0

    for sm in range(num_prints):
        img_imdjust = np.zeros((input_shape, input_shape, num_channels))
        img_imdj_n_denoi = np.zeros((input_shape, input_shape, num_channels))
        for ch in range(num_channels):
            img_ch = x_data[sm,:,:,ch]
            img_ch = np.uint8(cv2.normalize(np.float32(img_ch), None, 0, 255, cv2.NORM_MINMAX))
            img_chi = imadjust(img_ch)
            img_ch = cv2.fastNlMeansDenoising(img_chi, None, 30, 7, 21)
            img_imdjust[:,:,ch] = img_chi
            img_imdj_n_denoi[:,:,ch] = img_ch
        rgb = toimage(img_imdj_n_denoi)
        rgb = np.array(rgb)
        rgbi = toimage(img_imdjust)
        rgbi = np.array(rgbi) 

        plt.figure(1)
        plt.subplot(141)
        plt.imshow(img_imdjust[:,:,0], cmap='gray')
        plt.title('Band H')
        plt.grid(False)
        plt.subplot(142)
        plt.imshow(img_imdjust[:,:,1], cmap='gray')
        plt.title('Band J')
        plt.grid(False)
        plt.subplot(143)
        plt.imshow(img_imdjust[:,:,2], cmap='gray')
        plt.title('Band Y')
        plt.grid(False)
        plt.subplot(144)
        plt.imshow(rgbi)
        plt.title('Result RGB')
        plt.grid(False)
        plt.savefig('img_I_{}_{}.png'.format(sm, index))

        plt.figure(2)
        plt.subplot(141)
        plt.imshow(img_imdj_n_denoi[:,:,0], cmap='gray')
        plt.title('Band H')
        plt.grid(False)
        plt.subplot(142)
        plt.imshow(img_imdj_n_denoi[:,:,1], cmap='gray')
        plt.title('Band J')
        plt.grid(False)
        plt.subplot(143)
        plt.imshow(img_imdj_n_denoi[:,:,2], cmap='gray')
        plt.title('Band Y')
        plt.grid(False)
        plt.subplot(144)
        plt.imshow(rgb)
        plt.title('Result RGB')
        plt.grid(False)
        plt.savefig('img_I_D_{}_{}.png'.format(sm, index))

        for bu in range(num_prints*100):
            if os.path.exists(source+'./img_I_D_{}_{}.png'.format(sm, index)):
                shutil.move(source+'./img_I_D_{}_{}.png'.format(sm, index), dest1)
                counter = counter + 1
            if os.path.exists(source+'./img_I_{}_{}.png'.format(sm, index)):
                shutil.move(source+'./img_I_{}_{}.png'.format(sm, index), dest1)
                counter = counter + 1

    print("\n ** Done. %s files moved." % counter)
    return index

def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """
    Find the intersection of two arrays.
    Return the sorted, unique values that are in both of the input arrays.
    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. Will be flattened if not already 1D.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
        unique, incorrect results and out-of-bounds indices could result.
        Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.
        .. versionadded:: 1.15.0
    Returns
    -------
    intersect1d : ndarray
        Sorted 1D array of common and unique elements.
    comm1 : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    comm2 : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.
    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    Examples
    --------
    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])
    To intersect more than two arrays, use functools.reduce:
    >>> from functools import reduce
    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([3])
    To return the indices of the values common to the input arrays
    along with the intersected values:
    >>> x = np.array([1, 1, 2, 3, 4])
    >>> y = np.array([2, 1, 4, 6])
    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
    >>> x_ind, y_ind
    (array([0, 2, 4]), array([1, 0, 2]))
    >>> xy, x[x_ind], y[y_ind]
    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))
    """
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            ar1, ind1 = unique(ar1, return_index=True)
            ar2, ind2 = unique(ar2, return_index=True)
        else:
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = np.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d

def imadjust(src, tol=0.5, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    if (vin[1] - vin[0]) > 0:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    else:
        scale = 0
        
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    #####CRIAR BATCHES COM POTÊNCIAS DE 2 PARA RESOLVER O PROBLEMA DE 450 SAMPLES DA
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

#*******************************************************

def print_images(x_data, num_samples):
    count = 0
    Path('/home/kayque/LENSLOAD/').parent
    os.chdir('/home/kayque/LENSLOAD/')
    PATH = os.getcwd()

    for xe in range(0,num_samples,1):
        x_line = x_data[xe,:,:,:]
        print("x_line shape:", x_line.shape)
        #x_line = x_data[:][:][:][xe]
        rgb = toimage(x_line)
        rgb = np.array(rgb)
        im1 = Image.fromarray(rgb)
        #im1 = im1.resize((101,101), Image.ANTIALIAS)
        #cv2.resize(im1, (84, 84))
        im1.save("img_Y_%s.png" % xe)
        count = count + 1

    source = '/home/kayque/LENSLOAD/'
    print(' ** Does an old folder exists?')
    if os.path.exists(source+'lens_yes_2'):
        print(' ** Yes, it does! Trying to delete... ')
        shutil.rmtree(source+'lens_yes_2', ignore_errors=True)
        print(" ** Supposedly done. Checking if there's an RNCV folder...")
        os.mkdir('lens_yes_2')
        print(' ** Done!')
    else:
        print(" ** None found. Creating one.")
        os.mkdir('lens_yes_2')
        print(' ** Done!')

    dest1 = ('/home/kayque/LENSLOAD/lens_yes_2/')
    counter = 0

    for bu in range(0, num_samples, 1):
        if os.path.exists(source+'./img_Y_%s.png' % bu):
            shutil.move(source+'./img_Y_%s.png' % bu, dest1)
            counter = counter + 1

    print("\n ** Done. %s files moved." % counter)

def save_clue(x_data, y_data, TR, version, step, input_shape, nrows, ncols, index, channels):
    #fig = plt.figure(figsize=(10,3))
    #NN = x_data.shape[-1]
    #for i in range(NN):
        #plt.subplot(1,NN,i+1)
        #_ = plt.hist(np.clip(x_data[:,:,:,i].flatten(),-0.4e-10,2.5e-10),bins=256)
        #_ = plt.title(channels[i])
    #fig.savefig('histogram_for_normalization.png')

    figcount = index
    plt.figure()
    fig, axs = plt.subplots(nrows, ncols, figsize=(20,20))
    for i in range(nrows):
        for j in range(ncols):
            temp_image = toimage(np.array(x_data[figcount, :, :, :]))
            axs[i, j].imshow(temp_image)
            axs[i, j].set_title('Class: %s' % y_data[figcount])
            figcount = figcount + 1
    if figcount > len(y_data):
        index = 1
    else:
        index = figcount + 1
    plt.savefig("CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png". format(TR, version, step, input_shape, input_shape, index))
    print("CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png saved.". format(TR, version, step, input_shape, input_shape, index))

    #index = print_images_clecio_like(x_data=x_data, num_prints=10, num_channels=len(channels), input_shape=input_shape, index)
    return index

# Separa os conjuntos de treino, teste e validação.
def test_samples_balancer(y_data, x_data, vallim, train_size, percent_data, challenge):
    foldtimer = time.perf_counter()
    y_size = len(y_data)
    y_yes, y_no, y_excess = ([] for i in range(3))
    e_lente = 0
    n_lente = 0
    print(y_size)
    if challenge == 'challenge1':
        if train_size > 1600:
            print(' ** Using Temporary train_size')
            train_size = train_size/10
        else:
            print(' ** Using Regular train_size')

    for y in range(y_size):

        if y_data[y] == 1.0:
            # Pegamos uma quantidade de dados para treino e o que sobra vai para o excess e é usado para validação/teste
            e_lente += 1
            if len(y_yes) < (train_size * 5):
                # Armazenamos os índices
                y_yes = np.append(int(y), y_yes)
            else:
                y_excess = np.append(int(y), y_excess)
        else:
            n_lente += 1
            if len(y_no) < (train_size * 5):
                y_no = np.append(int(y), y_no)
            else:
                y_excess = np.append(int(y), y_excess)

    print(' -- Casos lente = ', e_lente)
    print(' -- Casos nao lente = ', n_lente)
    print(len(y_yes), len(y_no), len(y_excess))
    y_y = np.append(y_no, y_yes)
    np.random.shuffle(y_y)

    np.random.shuffle(y_excess)
    y_y = y_y.astype(int)
    y_excess = np.array(y_excess)
    y_excess = y_excess.astype(int)
    print(len(y_excess))

    # Define o tamanho do conjunto de validação, utilizando a variável vallim (nesse caso 2.000)
    y_val = y_data[y_excess[0:vallim]]
    x_val = x_data[y_excess[0:vallim]]

    y_test = y_data[y_excess[vallim:int(len(y_excess)*percent_data)]]
    x_test = x_data[y_excess[vallim:int(len(y_excess)*percent_data)]]
    print(' -- Number of test samples: ', len(x_test))

    # Preenchemos o y_data, usando os índices criados no y_y
    y_data = y_data[y_y]
    x_data = x_data[y_y]
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Function TIME: %.3f minutes.' % elaps)

    return [y_data, x_data, y_test, x_test, y_val, x_val]

# Randomiza os dados para divisão nas folds
def load_data_kfold(k, x_data, y_data):
    print('Preparing Folds')
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_data, y_data))
    return folds

def get_callbacks(name_weights, patience_lr, name_csv, tensorboard_path = None):
    mcp_save = ModelCheckpoint(name_weights, monitor="val_loss", verbose=1, save_best_only=True)
    csv_logger = CSVLogger(name_csv)
    #early_stopper = EarlyStopping(monitor='val_loss',
    #                        min_delta=0,
    #                        patience=10,
    #                        verbose=0, 
    #                        mode='auto')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4,
                                       mode='max', min_lr=0.1e-6)
    
    return [mcp_save, reduce_lr_loss]#, early_stopper]

def select_optimizer(optimizer, learning_rate):
    if optimizer == 'sgd':
        print('\n ** Usando otimizador: ', optimizer)
        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        print('\n ** Usando otimizador: ', optimizer)
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = optimizer
    return opt


# Gera a curva ROC
def roc_curve_calculate(y_test, x_test, x_test_vis, model, rede):
    print(' ** Roc Curve Calculating')
    print('\n ** Preds for: ', rede)
    probs = model.predict([x_test_vis, x_test])
    
    print(len(probs))
    print('\n ** probs_ensemble: ', probs)

    thres = int(len(y_test))
    threshold_v = np.linspace(1, 0, thres)

    for j in range(2):
        probsp = probs[:, j]
        y_new = y_test[:, j]
    
        tpr, fpr = ([] for i in range(2))

        for tt in range(len(threshold_v)):
            thresh = threshold_v[tt]
            tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
            for xz in range(len(probsp)):
                if probsp[xz] > thresh:
                    if y_new[xz] == 1:
                        tp_score = tp_score + 1
                    else:
                        fp_score = fp_score + 1
                else:
                    if y_new[xz] == 0:
                        tn_score = tn_score + 1
                    else:
                        fn_score = fn_score + 1
            tp_rate = tp_score / (tp_score + fn_score)
            fp_rate = fp_score / (fp_score + tn_score)
            tpr.append(tp_rate)
            fpr.append(fp_rate)
        
        if j == 0:
            tpr_vis = tpr
            fpr_vis = fpr
            auc_vis = metrics.auc(fpr, tpr)
            print('\n ** AUC_VIS (via metrics.auc): {}'.format(auc_vis))
        else:
            auc = metrics.auc(fpr, tpr)
            print('\n ** AUC (via metrics.auc): {}'.format(auc))

    return [tpr, fpr, tpr_vis, fpr_vis, auc, auc_vis, thres]

def roc_curves_sec(y_test, x_test, x_test_vis, models, model_list, version):
    print('Roc Curve Calculating')
    print('\n ** Models: ', model_list)
    # resnet = model[0]
    # efn = model[1]
    probas = []

    i = 0
    for mod in models:
        if i == 0:
            probas = mod.predict([x_test_vis, x_test])
            print(len(probas))
            i += 1
        else:
            probas = ((mod.predict([x_test_vis, x_test]))+probas)
            print(len(probas))
    probs = probas / len(model_list)
    print(len(probs))
    print('\n ** probs_ensemble: ', probs)

    thres = int(len(y_test))
    threshold_v = np.linspace(1, 0, thres)

    for j in range(2):
        probsp = probs[:, j]
        y_new = y_test[:, j]
    
        tpr, fpr = ([] for i in range(2))

        for tt in range(0, len(threshold_v), 1):
            thresh = threshold_v[tt]
            tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
            for xz in range(0, len(probsp), 1):
                if probsp[xz] > thresh:
                    if y_new[xz] == 1:
                        tp_score = tp_score + 1
                    else:
                        fp_score = fp_score + 1
                else:
                    if y_new[xz] == 0:
                        tn_score = tn_score + 1
                    else:
                        fn_score = fn_score + 1
            tp_rate = tp_score / (tp_score + fn_score)
            fp_rate = fp_score / (fp_score + tn_score)
            tpr.append(tp_rate)
            fpr.append(fp_rate)

        if j == 0:
            tpr_vis = tpr
            fpr_vis = fpr
            auc_vis = metrics.auc(fpr, tpr)
            print('\n ** AUC_VIS (via metrics.auc): {}'.format(auc_vis))
        else:
            auc = metrics.auc(fpr, tpr)
            print('\n ** AUC (via metrics.auc): {}'.format(auc))

    return [tpr, fpr, tpr_vis, fpr_vis, auc, auc_vis, thres]

def acc_score(acc0, history, val_acc0, loss0, val_loss0):
    print('\n ** Calculating acc0, val_acc0, loss0, val_loss0')
    acc0 = np.append(acc0, history.history['accuracy'])
    val_acc0 = np.append(val_acc0, history.history['val_accuracy'])
    loss0 = np.append(loss0, history.history['loss'])
    val_loss0 = np.append(val_loss0, history.history['val_loss'])
    print('\n ** Finished Calculating!')

    return [acc0, val_acc0, loss0, val_loss0]


def acc_score_ensemble(acc0, history, val_acc0, loss0, val_loss0):
    print('\n ** Calculating acc0, val_acc0, loss0, val_loss0')
    acc0 = np.append(acc0, history.history['accuracy'])
    val_acc0 = np.append(val_acc0, history.history['val_accuracy'])
    loss0 = np.append(loss0, history.history['loss'])
    val_loss0 = np.append(val_loss0, history.history['val_loss'])
    print('\n ** Finished Calculating!')

    return [acc0, val_acc0, loss0, val_loss0]

def FScore_curves(rede, model, x_test, y_test, batch_size):
    print(' ** F Scores Curve Calculating')
    print('\n ** Preds for: ', rede)
    probs = model.predict(x_test)

    probsp = probs[:, 1]
    print('\n ** probsp: ', probsp)
    print('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]
    print('\n ** y_new: ', y_new)
    thres = int(len(y_test))

    threshold_v = np.linspace(1, 0, thres)
    prec, rec = ([] for i in range(2))

    for tt in range(len(threshold_v)):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
            if probsp[xz] > thresh:
                if y_new[xz] == 1:
                    tp_score = tp_score + 1
                else:
                    fp_score = fp_score + 1
            else:
                if y_new[xz] == 0:
                    tn_score = tn_score + 1
                else:
                    fn_score = fn_score + 1

        try:
            precision = tp_score/(tp_score+fp_score)
        except:
            precision = 1
        try:
            recall = tp_score/(tp_score+fn_score)
        except:
            recall = 1
        prec.append(precision)
        rec.append(recall)
        
    print(' -- precision: ', precision)
    print(' -- recall: ', recall)

    f_1s, f_001s = [[] for i in range(2)]
    for fs in range(len(prec)):
        try:
            f_1 = 2 * (prec[fs] * rec[fs])/(prec[fs]+rec[fs])
        except:
            f_1 = 0.0
        try:
            f_001 = (1+0.001) * (prec[fs] * rec[fs])/(0.001*prec[fs]+rec[fs])
        except:
            f_001 = 0.0
        f_1s.append(f_1)
        f_001s.append(f_001)
    
    f_1_score = np.max(f_1s)
    f_001_score = np.max(f_001s)


    #f_1_score = get_classification_metric(y_test, probs, False)
    #f_1_score = sklearn.metrics.f1_score(y_test[:, 1], probsp)
    #f_1_score = fbeta(y_test[:, 1], probsp, 1, batch_size)
    #f_001_score = get_classification_metric(y_test, probs, True)
    #f_001_score = sklearn.metrics.fbeta_score(y_test[:, 1], probsp, beta=0.01)
    #f_001_score = fbeta(y_test[:, 1], probsp, 0.01, batch_size)

    print('\n ** F1 Score: {}, Fbeta Score: {}'.format(f_1_score, f_001_score))

    return [prec, rec, f_1_score, f_001_score, thres]

def fbeta(y_true, y_pred, beta, batch_size):
    TP = (K.sum((y_pred * y_true), axis=-1)) / batch_size
    FP = (K.sum(((1 - y_pred) * y_true), axis=-1)) / batch_size
    FN = (K.sum((y_pred * (1 - y_true)), axis=-1)) / batch_size
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fbeta = (1 + beta^2) * precision * recall / ( beta^2 * precision + recall)
    fbeta = 1 - K.mean(fbeta)
    return fbeta

def FScore_curves_ensemble(y_test, x_test, models, model_list, batch_size):
    print(' ** FScore Curves Ensemble')
    print(' ** Preds: ', model_list)
    probas = []

    i = 0
    for mod in models:
        if i == 0:
            probas = mod.predict(x_test)
            print(len(probas))
            i += 1
        else:
            probas = ((mod.predict(x_test))+probas)
            print(len(probas))
    probs = probas / len(model_list)
    print(len(probs))
    print('\n ** probs_ensemble: ', probs)

    probsp = probs[:, 1]
    # print('\n ** probsp: ', probsp)
    # print('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]
    #print('\n ** y_new: ', y_new)
    thres = int(len(y_test))

    threshold_v = np.linspace(1, 0, thres)
    prec, rec = ([] for i in range(2))

    for tt in range(0, len(threshold_v), 1):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(0, len(probsp), 1):
            if probsp[xz] > thresh:
                if y_new[xz] == 1:
                    tp_score = tp_score + 1
                else:
                    fp_score = fp_score + 1
            else:
                if y_new[xz] == 0:
                    tn_score = tn_score + 1
                else:
                    fn_score = fn_score + 1
                    
        try:
            precision = tp_score/(tp_score+fp_score)
        except:
            precision = 1
        try:
            recall = tp_score/(tp_score+fn_score)
        except:
            recall = 1
        prec.append(precision)
        rec.append(recall)

    f_1s, f_001s = [[] for i in range(2)]
    for fs in range(len(prec)):
        try:
            f_1 = 2 * (prec[fs] * rec[fs])/(prec[fs]+rec[fs])
        except:
            f_1 = 0.0
        try:
            f_001 = (1+0.001) * (prec[fs] * rec[fs])/(0.001*prec[fs]+rec[fs])
        except:
            f_001 = 0.0
    
    f_1_score = np.max(f_1s)
    f_001_score = np.max(f_001s)

    #f_1_score = sklearn.metrics.f1_score(y_test[:, 1], np.around(probsp))
    #f_001_score = sklearn.metrics.fbeta_score(y_test[:, 1], np.around(probsp), beta=0.01)
    #f_1_score = get_classification_metric(y_test, probs, False)
    #f_001_score = get_classification_metric(y_test, probs, True)
    #f_1_score = fbeta(y_test[:, 1], probsp, 1, batch_size)
    #f_001_score = fbeta(y_test[:, 1], probsp, 0.01, batch_size)

    print('\n ** F1 Score: {}, Fbeta Score: {}'.format(f_1_score, f_001_score))

    return [prec, rec, f_1_score, f_001_score, thres]

def select_optimizer(optimizer, learning_rate):
    if optimizer == 'sgd':
        print('\n ** Usando otimizador: ', optimizer)
        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    else:
        print('\n ** Usando otimizador: ', optimizer)
        opt = optimizer
        #opt = Adam(learning_rate=learning_rate)
    return opt

def get_model_roulette(mod, img_shape, img_input, weights):
    n_model = []
    if mod == 'resnet50':
        resnet_depth = 50
        n_model = model_lib.get_model_resnet(img_shape, img_input, weights, resnet_depth)
    if mod == 'resnet101':
        resnet_depth = 101
        n_model = model_lib.get_model_resnet(img_shape, img_input, weights, resnet_depth)
    if mod == 'resnet152':
        resnet_depth = 152
        n_model = model_lib.get_model_resnet(img_shape, img_input, weights, resnet_depth)
    if mod == 'effnet_B0':
        effnet_version = 'B0'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'effnet_B1':
        effnet_version = 'B1'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'effnet_B2':
        effnet_version = 'B2'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'effnet_B3':
        effnet_version = 'B3'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'effnet_B4':
        effnet_version = 'B4'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'effnet_B5':
        effnet_version = 'B5'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'effnet_B6':
        effnet_version = 'B6'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'effnet_B7':
        effnet_version = 'B7'
        n_model = model_lib.get_model_effnet(img_shape, img_input, weights, effnet_version)
    if mod == 'inceptionV2':
        version = 'V2'
        n_model = model_lib.get_model_inception(img_shape, img_input, weights, version)
    if mod == 'inceptionV3':
        version = 'V3'
        n_model = model_lib.get_model_inception(img_shape, img_input, weights, version)
    if mod == 'xception':
        n_model = model_lib.get_model_xception(img_shape, img_input, weights)

    return n_model

def normalize(X, num_channels, apply_log, vis):
    pbar = Progbar(num_channels - 1)
    for k in range(num_channels):
        vmax = np.max(X[:,:,:,k])
        vmin = np.min(X[:,:,:,k]) 
        #vmax = vmin + 1.0
        print(vmin, vmax)
        X[:,:,:,k] = np.clip(X[:,:,:,k], vmin, vmax)
        X[:,:,:,k] = (X[:,:,:,k] - vmin) / (vmax - vmin)
        print(' -- after_clipping...')
        print(np.min(X[:,:,:,k]), np.max(X[:,:,:,k]))
        if apply_log:
            if not vis:
                X[:,:,:,k] = np.log10(X[:,:,:,k].astype(np.float16) + 1e-8)
            else:
                X[:,:,:,k] = np.log10(X[:,:,:,k].astype(np.float16) - 1e-8)
        X[:,:,:,k][np.where(np.isnan(X[:,:,:,k]))] = 0
        print(' -- after_log:')
        print(np.min(X[:,:,:,k]), np.max(X[:,:,:,k]))
        print(X.shape)

        if k != 0:
            pbar.update(k)
    return X

def normalize2(X, num_channels, channels, apply_log, vis):
    pbar = Progbar(num_channels - 1)
    for k in range(num_channels):
        vmax = np.percentile(X[:,:,:,k], 98)
        vmin = -np.percentile(-X[:,:,:,k], 99.9) 
        graphs.histo(X[:,:,:,k], vmin, vmax, True, channels[k])
        #vmax = vmin + 1.0
        print(vmin, vmax)
        X[:,:,:,k] = np.clip(X[:,:,:,k], vmin, vmax)
        X[:,:,:,k] = (X[:,:,:,k] - vmin) / (vmax - vmin)
        print(' -- after_clipping...')
        print(np.min(X[:,:,:,k]), np.max(X[:,:,:,k]))
        graphs.histo(X[:,:,:,k], vmin, vmax, False, channels[k])
        if apply_log:
            if not vis:
                X[:,:,:,k] = np.log10(X[:,:,:,k].astype(np.float16) + 1e-8)
            else:
                X[:,:,:,k] = np.log10(X[:,:,:,k].astype(np.float16) - 1e-8)
            print(' -- after_log:')
            print(np.min(X[:,:,:,k]), np.max(X[:,:,:,k]))
        X[:,:,:,k][np.where(np.isnan(X[:,:,:,k]))] = 0
        print(X.shape)

        if k != 0:
            pbar.update(k)
    return X

def get_classification_metric(testy, probs, beta, arg=0.01):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(testy, probs[:,1])
    # convert to f score
    if beta:
        fscore = ((1*(arg**2) * precision * recall) / (precision + recall))
    else:
        fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    return fscore[ix]

def ElapsedTime(t0):
    tnow = time.time()
    hh = 0
    mm = 0
    ss = 0
    tep = tnow-t0
    hh = tep//3600
    tep = tep - (hh*3600)
    mm = tep//60
    ss = tep - (mm*60)
    
    return "%02d:%02d:%02d" % (hh,mm,ss)

def delete_weights(ix, version):
    weicounter = 0
    for epoch in range(501):
        if epoch < 10:
            # Números menores que 10
            if os.path.exists('./Train_model_weights_{}_0{}_{}.h5'. format(ix, epoch, version)):
                os.remove('./Train_model_weights_{}_0{}_{}.h5'. format(ix, epoch, version))
                weicounter = weicounter + 1
            if os.path.exists('./Train_model_weights_{}_0{}_{}_Backup.h5'. format(ix, epoch, version)):
                os.remove('./Train_model_weights_{}_0{}_{}_Backup.h5'. format(ix, epoch, version))
                weicounter = weicounter + 1
        else:
            # Números maiores que 10
            if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(ix, epoch, version)):
                os.remove('./Train_model_weights_{}_{}_{}.h5'. format(ix, epoch, version))
                weicounter = weicounter + 1
            if os.path.exists('./Train_model_weights_{}_{}_{}_Backup.h5'. format(ix, epoch, version)):
                if epoch != 50:
                    os.remove('./Train_model_weights_{}_{}_{}_Backup.h5'. format(ix, epoch, version))
                    weicounter = weicounter + 1
    print(' ** %s Train model weights deleted.' % weicounter)

MAP_INTERPOLATION_TO_ORDER = {
    "nearest": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3,
}

def center_crop_and_resize(image, image_size, crop_padding=32, interpolation="bicubic"):
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]

    padded_center_crop_size = int(
        (image_size / (image_size + crop_padding)) * min(h, w)
    )
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[
                 offset_height: padded_center_crop_size + offset_height,
                 offset_width: padded_center_crop_size + offset_width,
                 ]
    resized_image = resize(
        image_crop,
        (image_size, image_size),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True,
    )

    return resized_image