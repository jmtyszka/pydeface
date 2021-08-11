#!/usr/bin/env python
#
# Copyright (C) 2013-2015 Russell Poldrack <poldrack@stanford.edu>
# some portions borrowed from https://github.com/mwaskom/lyman/blob/master/setup.py


descr = """pydeface: deidentify structural MRI data by voxelizing the face"""

import os
from setuptools import setup
import glob

DISTNAME="pydeface"
DESCRIPTION=descr
MAINTAINER='Mike Tyszka'
MAINTAINER_EMAIL='jmt@caltech.edu'
LICENSE='MIT'
URL='http://evendim.sites.caltech.edu'
DOWNLOAD_URL='https://github.com/jmtyszka/pydeface/'
VERSION='2020.4.3'

def check_dependencies():

    # Just make sure dependencies exist, I haven't rigorously
    # tested what the minimal versions that will work are
    needed_deps = ["numpy", "nibabel", "nipype"]
    missing_deps = []
    for dep in needed_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        raise ImportError("Missing dependencies: %s" % missing_deps)

if __name__ == "__main__":

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    import sys
    if not (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands',
                            '--version',
                            'egg_info',
                            'clean'))):
        check_dependencies()

    datafiles = {'pydeface': ['data/ConteCore2_50_T1w_2mm.nii.gz',
                              'data/ConteCore2_50_T1w_2mm_deface_mask.nii.gz',
                              'ident.mat']}

    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        url=URL,
        download_url=DOWNLOAD_URL,
        packages=['pydeface'],
        package_data  = datafiles,
        scripts=['scripts/pydeface.py'],
        classifiers=['Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'License :: OSI Approved :: BSD License',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        install_requires=['nipype',
                          'numpy'],
    )
