import numpy as np
import os
import argparse
import pypcd
import csv
from tqdm import tqdm
import re
import warnings
import struct
import lzf


numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

def parse_header(lines):
        """ Parse header of PCD files.
        """
        metadata = {}
        for ln in lines:
            if ln.startswith('#') or len(ln) < 2:
                continue
            match = re.match('(\w+)\s+([\w\s\.]+)', ln)
            if not match:
                warnings.warn("warning: can't understand line: %s" % ln)
                continue
            key, value = match.group(1).lower(), match.group(2)
            if key == 'version':
                metadata[key] = value
            elif key in ('fields', 'type'):
                metadata[key] = value.split()
            elif key in ('size', 'count'):
                metadata[key] = map(int, value.split())
            elif key in ('width', 'height', 'points'):
                metadata[key] = int(value)
            elif key == 'viewpoint':
                metadata[key] = map(float, value.split())
            elif key == 'data':
                metadata[key] = value.strip().lower()
            # TODO apparently count is not required?
        # add some reasonable defaults
        if 'count' not in metadata:
            metadata['count'] = [1]*len(metadata['fields'])
        if 'viewpoint' not in metadata:
            metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if 'version' not in metadata:
            metadata['version'] = '.7'
        return metadata
def parse_ascii_pc_data(f, dtype, metadata):
        """ Use numpy to parse ascii pointcloud data.
        """
        return np.loadtxt(f, dtype=dtype, delimiter=' ')  
def parse_binary_pc_data(f, dtype, metadata):
        rowstep = metadata['points']*dtype.itemsize
        # for some reason pcl adds empty space at the end of files
        buf = f.read(rowstep)
        return np.fromstring(buf, dtype=dtype)
def parse_binary_compressed_pc_data(self,f, dtype, metadata):
        """ Parse lzf-compressed data.
        Format is undocumented but seems to be:
        - compressed size of data (uint32)
        - uncompressed size of data (uint32)
        - compressed data
        - junk
        """
        fmt = 'II'
        compressed_size, uncompressed_size =\
            struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        compressed_data = f.read(compressed_size)
        # TODO what to use as second argument? if buf is None
        # (compressed > uncompressed)
        # should we read buf as raw binary?
        buf = lzf.decompress(compressed_data, uncompressed_size)
        if len(buf) != uncompressed_size:
            raise IOError('Error decompressing data')
        # the data is stored field-by-field
        pc_data = np.zeros(metadata['width'], dtype=dtype)
        ix = 0
        for dti in range(len(dtype)):
            dt = dtype[dti]
            bytes = dt.itemsize * metadata['width']
            column = np.fromstring(buf[ix:(ix+bytes)], dt)
            pc_data[dtype.names[dti]] = column
            ix += bytes
        return pc_data
    
def _build_dtype(metadata):
        """ Build numpy structured array dtype from pcl metadata.

        Note that fields with count > 1 are 'flattened' by creating multiple
        single-count fields.

        *TODO* allow 'proper' multi-count fields.
        """
        fieldnames = []
        typenames = []
        for f, c, t, s in zip(metadata['fields'],
                            metadata['count'],
                            metadata['type'],
                            metadata['size']):
            np_type = pcd_type_to_numpy_type[(t, s)]
            if c == 1:
                fieldnames.append(f)
                typenames.append(np_type)
            else:
                fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
                typenames.extend([np_type]*c)
        # dtype = np.dtype(zip(fieldnames, typenames))
        # dtype = tuple(typenames)
        dtype = np.dtype({'names':tuple(fieldnames),
                 'formats':tuple(typenames)})
        return dtype  
def pcd_reader(pcd_path):
        """ Parse pointcloud coming from file object f
        """
        with open(pcd_path, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip().decode('utf-8')
                header.append(ln)
                if ln.startswith('DATA'):
                    metadata = parse_header(header)
                    dtype = _build_dtype(metadata)
                    break
            if metadata['data'] == 'ascii':
                pc_data = parse_ascii_pc_data(f, dtype, metadata)
            elif metadata['data'] == 'binary':
                pc_data = parse_binary_pc_data(f, dtype, metadata)
            elif metadata['data'] == 'binary_compressed':
                pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
        
        return pc_data
def main():
    ## Add parser
    parser = argparse.ArgumentParser(description="Convert .pcd to .bin")
    parser.add_argument(
        "--pcd_path",
        help=".pcd file path.",
        type=str,
        default='/home/jw11/bagKits/01-Lidar_2023-02-26-12-01-30/pcd_out/splice3lidar'
    )
    parser.add_argument(
        "--bin_path",
        help=".bin file path.",
        type=str,
        default="/home/jw11/bagKits/01-Lidar_2023-02-26-12-01-30/pcd_out/bin_file"
    )
    parser.add_argument(
        "--file_name",
        help="File name.",
        type=str,
        default="shanhai"
    )
    args = parser.parse_args()

    ## Find all pcd files
    pcd_files = []
    for (path, dir, files) in os.walk(args.pcd_path):
        for filename in files:
            # print(filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.pcd':
                pcd_files.append(path + "/" + filename)

    ## Sort pcd files by file name
    pcd_files.sort()   
    print("Finish to load point clouds!")

    ## Make bin_path directory
    try:
        if not (os.path.isdir(args.bin_path)):
            os.makedirs(os.path.join(args.bin_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print ("Failed to create directory!!!!!")
            raise

    ## Generate csv meta file
    csv_file_path = os.path.join(args.bin_path, "meta.csv")
    csv_file = open(csv_file_path, "w")
    meta_file = csv.writer(
        csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    ## Write csv meta file header
    meta_file.writerow(
        [
            "pcd file name",
            "bin file name",
        ]
    )
    print("Finish to generate csv meta file")

    ## Converting Process
    print("Converting Start!")
    seq = 0
    for pcd_file in tqdm(pcd_files):
        ## Get pcd file
        pc = pcd_reader(pcd_file)

        ## Generate bin file name
        bin_file_name = "{}_{:05d}.bin".format(args.file_name, seq)
        bin_file_path = os.path.join(args.bin_path, bin_file_name)
        
        ## Get data from pcd (x, y, z, intensity, ring, time)
        np_x = (np.array(pc['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc['intensity'], dtype=np.float32)).astype(np.float32)/256
        # np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
        # np_t = (np.array(pc.pc_data['time'], dtype=np.float32)).astype(np.float32)

        ## Stack all data    
        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))

        ## Save bin file                                    
        points_32.tofile(bin_file_path)

        ## Write csv meta file
        meta_file.writerow(
            [os.path.split(pcd_file)[-1], bin_file_name]
        )

        seq = seq + 1
    
if __name__ == "__main__":
    main()
