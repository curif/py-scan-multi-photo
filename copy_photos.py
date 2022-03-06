import hashlib
import argparse
import os
import shutil

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

parser = argparse.ArgumentParser(description='copy processed photos where the files are "XXX_N.png@ where XXX is a prefix, N is a number and png is the filte type')
parser.add_argument('--input_prefix', help="name prefix for processed photos", default='ROI_')
parser.add_argument('--file_type', help="png/jpeg/jpg/etc", default='png')
parser.add_argument('--dest_path', help="path where to copy to", default='./')
parser.add_argument('--output_prefix', help="IMG, photo or what you want for the start of the name of the copied file", default='IMG')
parser.add_argument('--filter', help='comma separated list of numbers to filter, like 7,10,11,12', default='')
args = parser.parse_args()

expected_files = [
    '{}{}.{}'.format(args.input_prefix, number, args.file_type)
    for number in args.filter.split(',')
]
print(expected_files)
obj = os.scandir('./')
for entry in obj :
    if entry.is_file()\
        and entry.name.startswith(args.input_prefix)\
        and entry.name.endswith(args.file_type)\
        and (len(expected_files)==0 or entry.name in expected_files):
            file_hash = md5(entry.name)[-10:]
            dest = os.path.join(args.dest_path, '{}-{}.{}'.format(args.output_prefix, file_hash, args.file_type))
            print('{} -> {}'.format(entry.name, dest))
            shutil.copy(entry.name, dest)
