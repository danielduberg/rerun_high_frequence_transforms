import numpy as np
import sys
from typing import Iterable, List, Optional

INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
FLOAT32 = 7
FLOAT64 = 8

_DATATYPES = {}
_DATATYPES[INT8] = np.dtype(np.int8)
_DATATYPES[UINT8] = np.dtype(np.uint8)
_DATATYPES[INT16] = np.dtype(np.int16)
_DATATYPES[UINT16] = np.dtype(np.uint16)
_DATATYPES[INT32] = np.dtype(np.int32)
_DATATYPES[UINT32] = np.dtype(np.uint32)
_DATATYPES[FLOAT32] = np.dtype(np.float32)
_DATATYPES[FLOAT64] = np.dtype(np.float64)

DUMMY_FIELD_PREFIX = 'unnamed_field'


def _get_fields_and_offsets(dt, offset=0):
    # counts up elements in subarrays, including nested subarrays, and returns
    # base dtype and count
    def count_elem(dt):
        count = 1
        while dt.shape != ():
            for size in dt.shape:
                count *= size
            dt = dt.base
        return dt, count

    fields = []
    for name in dt.names:
        field = dt.fields[name]
        f_dt, f_offset = field[0], field[1]
        f_dt, n = count_elem(f_dt)

        if f_dt.names is None:
            fields.append((np.dtype((f_dt, (n,))), n, f_offset + offset))
        else:
            subfields = _get_fields_and_offsets(f_dt, f_offset + offset)
            size = f_dt.itemsize

            for i in range(n):
                if i == 0:
                    # optimization: avoid list comprehension if no subarray
                    fields.extend(subfields)
                else:
                    fields.extend([(d, c, o + i*size)
                                  for d, c, o in subfields])
    return fields


def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')

    fields = _get_fields_and_offsets(arr.dtype)
    n_fields = len(fields)
    if n_fields == 0 and dtype is None:
        raise ValueError('arr has no fields. Unable to guess dtype')
    elif n_fields == 0:
        # too many bugs elsewhere for this to work now
        raise NotImplementedError('arr with no fields is not supported')

    dts, counts, offsets = zip(*fields)
    names = ['f{}'.format(n) for n in range(n_fields)]

    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        out_dtype = dtype

    # Use a series of views and casts to convert to an unstructured array:

    # first view using flattened fields (doesn't work for object arrays)
    # Note: dts may include a shape for subarrays
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': arr.dtype.itemsize})
    arr = arr.view(flattened_fields)

    # next cast to a packed format with all fields converted to new dtype
    packed_fields = np.dtype({'names': names,
                              'formats': [(out_dtype, dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)

    # finally is it safe to view the packed fields as the unstructured type
    return arr.view((out_dtype, (sum(counts),)))


def unstructured_to_structured(arr, dtype=None, names=None, align=False,
                               copy=False, casting='unsafe'):
    if arr.shape == ():
        raise ValueError('arr must have at least one dimension')
    n_elem = arr.shape[-1]
    if n_elem == 0:
        # too many bugs elsewhere for this to work now
        raise NotImplementedError('last axis with size 0 is not supported')

    if dtype is None:
        if names is None:
            names = ['f{}'.format(n) for n in range(n_elem)]
        out_dtype = np.dtype([(n, arr.dtype) for n in names], align=align)
        fields = _get_fields_and_offsets(out_dtype)
        dts, counts, offsets = zip(*fields)
    else:
        if names is not None:
            raise ValueError("don't supply both dtype and names")
        # sanity check of the input dtype
        fields = _get_fields_and_offsets(dtype)
        if len(fields) == 0:
            dts, counts, offsets = [], [], []
        else:
            dts, counts, offsets = zip(*fields)

        if n_elem != sum(counts):
            raise ValueError('The length of the last dimension of arr must '
                             'be equal to the number of fields in dtype')
        out_dtype = dtype
        if align and not out_dtype.isalignedstruct:
            raise ValueError('align was True but dtype is not aligned')

    names = ['f{}'.format(n) for n in range(len(fields))]

    # Use a series of views and casts to convert to a structured array:

    # first view as a packed structured array of one dtype
    packed_fields = np.dtype({'names': names,
                              'formats': [(arr.dtype, dt.shape) for dt in dts]})
    arr = np.ascontiguousarray(arr).view(packed_fields)

    # next cast to an unpacked but flattened format with varied dtypes
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': out_dtype.itemsize})
    arr = arr.astype(flattened_fields, copy=copy, casting=casting)

    # finally view as the final nested dtype and remove the last axis
    return arr.view(out_dtype)[..., 0]


def read_points(
        cloud,
        field_names: Optional[List[str]] = None,
        skip_nans: bool = False,
        uvs: Optional[Iterable] = None,
        reshape_organized_cloud: bool = False) -> np.ndarray:
    """
    Read points from a sensor_msgs.PointCloud2 message.

    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param skip_nans: If True, then don't return any point with a NaN value.
                      (Type: Bool, Default: False)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: None)
    :param reshape_organized_cloud: Returns the array as an 2D organized point cloud if set.
    :return: Structured NumPy array containing all points.
    """
    # assert isinstance(cloud, PointCloud2), \
    #     'Cloud is not a sensor_msgs.msg.PointCloud2'

    # Cast bytes to numpy array
    points = np.ndarray(
        shape=(cloud.width * cloud.height, ),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data)

    # Keep only the requested fields
    if field_names is not None:
        assert all(field_name in points.dtype.names for field_name in field_names), \
            'Requests field is not in the fields of the PointCloud!'
        # Mask fields
        points = points[list(field_names)]

    # Swap array if byte order does not match
    if bool(sys.byteorder != 'little') != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)

    # Check if we want to drop points with nan values
    if skip_nans and not cloud.is_dense:
        # Init mask which selects all points
        not_nan_mask = np.ones(len(points), dtype=bool)
        for field_name in points.dtype.names:
            # Only keep points without any non values in the mask
            not_nan_mask = np.logical_and(
                not_nan_mask, ~np.isnan(points[field_name]))
        # Select these points
        points = points[not_nan_mask]

    # Select points indexed by the uvs field
    if uvs is not None:
        # Don't convert to numpy array if it is already one
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        # Index requested points
        points = points[uvs]

    # Cast into 2d array if cloud is 'organized'
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)

    return points


def read_points_numpy(
        cloud,
        field_names: Optional[List[str]] = None,
        skip_nans: bool = False,
        uvs: Optional[Iterable] = None,
        reshape_organized_cloud: bool = False) -> np.ndarray:
    """
    Read equally typed fields from sensor_msgs.PointCloud2 message as a unstructured numpy array.

    This method is better suited if one wants to perform math operations
    on e.g. all x,y,z fields.
    But it is limited to fields with the same dtype as unstructured numpy arrays
    only contain one dtype.

    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param skip_nans: If True, then don't return any point with a NaN value.
                      (Type: Bool, Default: False)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: None)
    :param reshape_organized_cloud: Returns the array as an 2D organized point cloud if set.
    :return: Numpy array containing all points.
    """
    assert all(cloud.fields[0].datatype == field.datatype for field in cloud.fields[1:]
               if field_names is None or field.name in field_names), \
        'All fields need to have the same datatype. Use `read_points()` otherwise.'
    structured_numpy_array = read_points(
        cloud, field_names, skip_nans, uvs, reshape_organized_cloud)
    return structured_to_unstructured(structured_numpy_array)


def dtype_from_fields(fields, point_step: Optional[int] = None) -> np.dtype:
    """
    Convert a Iterable of sensor_msgs.msg.PointField messages to a np.dtype.

    :param fields: The point cloud fields.
                   (Type: iterable of sensor_msgs.msg.PointField)
    :param point_step: Point step size in bytes. Calculated from the given fields by default.
                       (Type: optional of integer)
    :returns: NumPy datatype
    """
    # Create a lists containing the names, offsets and datatypes of all fields
    field_names = []
    field_offsets = []
    field_datatypes = []
    for i, field in enumerate(fields):
        # Datatype as numpy datatype
        datatype = _DATATYPES[field.datatype]
        # Name field
        if field.name == '':
            name = f'{DUMMY_FIELD_PREFIX}_{i}'
        else:
            name = field.name
        # Handle fields with count > 1 by creating subfields with a suffix consiting
        # of "_" followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            if field.count > 1:
                subfield_name = f'{name}_{a}'
            else:
                subfield_name = name
            assert subfield_name not in field_names, 'Duplicate field names are not allowed!'
            field_names.append(subfield_name)
            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)

    # Create dtype
    dtype_dict = {
        'names': field_names,
        'formats': field_datatypes,
        'offsets': field_offsets
    }
    if point_step is not None:
        dtype_dict['itemsize'] = point_step
    return np.dtype(dtype_dict)
