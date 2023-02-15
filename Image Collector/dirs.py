import requests
import sys
import os
import inspect
import sys

def current():
	return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def parent(current):
	for i in range(len(current)):
		if (current[-i] == '\\' and i != 0) or (current[-i] == '/' and i != 0):  # in python 3.7
			pdir = current[:-i+1]
			# print(pdir)
			return pdir


def join(current_path, dirs, insert_path=False):
	paths = []
	if isinstance(dirs, list):
		for dir in dirs:
			paths.append(os.path.join(current_path, dir))
		if insert_path:
			insert(paths)
		return paths
	
	paths = os.path.join(current_path, dirs)
	if insert_path:
		insert(paths)

	return paths


def insert(path):
	sys.path.insert(1, path)


def files(path):

	fnames = []
	dnames = []

	for f in os.listdir(path):
		if os.path.isfile(join(path, f)):
			fnames.append(f)
		else:
			if f[0] != '.' and f[0] != '_':
				dnames.append(f)

	return fnames, dnames


def tmp_files_dir():
	''' Temporary direction to files dir, in the future it will be by choice '''
	currentdir = current()
	parentdir = parent(currentdir)
	_, dnames = files(parentdir)
	path = join(parentdir, dnames[1])
	_, dnames = files(path)
	path = join(path, dnames[2])
	# path = os.path.join(path, dnames[1])
	_, dnames = files(path)
	path = join(path, dnames[0])
	# path = os.path.join(path, dnames[1])
	f, d = files(path)
	# print(d)
	return path, f, d


def exp_dir():
	currentdir = current()
	parentdir = parent(currentdir)
	_, dnames = files(parentdir)

	exp_folder = []
	for i, dname in enumerate(dnames):
		if dname[:3] == 'Exp':
			exp_folder = join(parentdir, dname)
			break
	_, exp_dnames = files(exp_folder) # Gets Expiremnts dirs names

	exp_paths = [] # Expirements dirs paths
	exp_paths = join(exp_folder, exp_dnames)
	
	return exp_dnames, exp_paths
			 
# exp_dnames, exp_paths = exp_dir()
# print(exp_dnames, exp_paths)