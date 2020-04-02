#!/usr/bin/env python3
"""
deface an image using FSL
USAGE:  deface -i <filename to deface> -o [output filename]

## Copyright 2011, Russell Poldrack. All rights reserved.

## Redistribution and use in source and binary forms, with or without modification, are
## permitted provided that the following conditions are met:

##    1. Redistributions of source code must retain the above copyright notice, this list of
##       conditions and the following disclaimer.

##    2. Redistributions in binary form must reproduce the above copyright notice, this list
##       of conditions and the following disclaimer in the documentation and/or other materials
##       provided with the distribution.

## THIS SOFTWARE IS PROVIDED BY RUSSELL POLDRACK ``AS IS'' AND ANY EXPRESS OR IMPLIED
## WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
## FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL RUSSELL POLDRACK OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
## ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
## NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import os
import sys
import tempfile
import subprocess
import argparse
import nibabel as nib
import shutil
from scipy import ndimage as nd
from nipype.interfaces import fsl
from pkg_resources import resource_filename, Requirement


def run_shell_cmd(cmd, cwd=[]):
    """
    Run a command in the shell using Popen
    """
    if cwd:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, cwd=cwd)
    else:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.strip())

    process.wait()


def main():

    # T1w template and face mask locations
    T1w_template = resource_filename(Requirement.parse("pydeface"), 'pydeface/data/T1w_template.nii.gz')
    facemask = resource_filename(Requirement.parse("pydeface"), "pydeface/data/facemask.nii.gz")

    try:
        assert os.path.exists(T1w_template)
    except:
        raise Exception('*** Missing template : %s'%T1w_template)

    try:
        assert os.path.exists(facemask)
    except:
        raise Exception('*** Missing facemask : %s'%facemask)

    # Check that FSLDIR is set
    if 'FSLDIR' in os.environ:
        FSLDIR=os.environ['FSLDIR']
    else:
        print('FSL must be installed and FSLDIR environment variable must be defined')
        sys.exit(2)

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Remove facial structure from MRI images')
    parser.add_argument('-i', '--infile', required=True,
                        help='T1w input image (Nifti or Minc')
    parser.add_argument('-o', '--outfile', required=False,
                        help='Defaced output image [<infile>_defaced.<ext>]')
    parser.add_argument('-s', '--scalefactor', required=False,
                        help='Voxelation scale factor [8.0]')
    parser.add_argument('-r', '--replace', action='store_true', default=False,
                        help='Deface image in place and backup original [False]')

    # Parse command line arguments
    args = parser.parse_args()
    in_fname = args.infile

    print('Input faced image    : {}'.format(in_fname))

    # Save full image extension
    # Handle double-extension for compressed Nifti files
    if in_fname.endswith('.nii.gz'):
        img_ext = '.nii.gz'
    else:
        in_stub, img_ext = os.path.splitext(in_fname)

    # Output defaced image filename
    if args.outfile:
        out_fname = args.outfile
    else:
        out_fname = in_fname.replace(img_ext, '_defaced' + img_ext)

    #

    # Protect existing output file
    if os.path.isfile(out_fname):
        print('{} already exists - remove it first'.format(out_fname))
        sys.exit(1)

    print('Output defaced image : {}'.format(out_fname))

    if args.scalefactor:
        vox_sf = float(args.scalefactor)
    else:
        vox_sf = 8.0

    # Temporary template to individual affine transform matrix
    _, tmp_temp2ind_mat_fname = tempfile.mkstemp()
    tmp_temp2ind_mat_fname += '.mat'

    # Temporary face mask in individual space
    _, tmp_indmask_fname = tempfile.mkstemp()
    tmp_indmask_fname += '.nii.gz'

    # Temporary registration output results
    _, tmp_img_fname = tempfile.mkstemp()
    _, tmp_mat_fname = tempfile.mkstemp()

    print('Defacing {}'.format(in_fname))

    # Register template to infile
    print('Registering template to individual space')
    flirt = fsl.FLIRT()
    flirt.inputs.cost_func='mutualinfo'
    flirt.inputs.in_file = T1w_template
    flirt.inputs.out_matrix_file = tmp_temp2ind_mat_fname
    flirt.inputs.reference = in_fname
    flirt.inputs.out_file = tmp_img_fname
    flirt.run()

    # Affine transform facemask to infile
    print('Resampling face mask to individual space')
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = facemask
    flirt.inputs.in_matrix_file = tmp_temp2ind_mat_fname
    flirt.inputs.apply_xfm = True
    flirt.inputs.reference = in_fname
    flirt.inputs.out_file = tmp_indmask_fname
    flirt.inputs.out_matrix_file = tmp_mat_fname
    flirt.run()

    # Load input image
    print('Loading {}'.format(in_fname))
    in_nii = nib.load(in_fname)
    in_img = in_nii.get_data()

    # Voxelate input image
    # 1. Cubic downsample
    # 2. Nearest neighbor upsample
    print('Voxelating faced image : scale factor %0.1f' % vox_sf)

    print('  Spline downsampling')
    in_vox_img = nd.interpolation.zoom(in_img, zoom=1.0/vox_sf, order=3)

    print('  Nearest neighbor upsampling')
    in_vox_img = nd.interpolation.zoom(in_vox_img, zoom=vox_sf, order=0)

    # Load individual space face mask
    indmask_nii = nib.load(tmp_indmask_fname)
    indmask_img = indmask_nii.get_data()

    # Replace face area with voxelated version
    # Note that the face mask is 0 in the face region, 1 elsewhere.
    print('Anonymizing face area')
    out_img = in_img * indmask_img + in_vox_img * (1 - indmask_img)

    # Save defaced image
    print('Saving defaced image to {}'.format(out_fname))

    if '.nii' in out_fname:
        outfile_obj = nib.Nifti1Image(out_img, in_nii.get_affine(), in_nii.get_header())
    elif '.mgz' in out_fname:
        outfile_obj = nib.MincImage(out_img, in_nii.get_affine(), in_nii.get_header())
    else:
        print('* Unknown output format extension {} - exiting'.format(img_ext))
        sys.exit(1)

    # Export defaced voxelated image. Output format handled by nibabel
    outfile_obj.to_filename(out_fname)

    # Handle replacement with backup option
    if args.replace:

        out_fname = in_fname

        # Backup original image
        bak_fname = in_fname.replace(img_ext, '_faced' + img_ext)
        print('Backup faced image   : {}'.format(bak_fname))

        # Backup original
        try:
            shutil.move(in_fname, bak_fname)
        except IOError as e:
            print('I/O error({0}): {1}'.format(e.errno, e.strerror))
            print('Could not backup original image - exiting')
            sys.exit(1)

        # Replace original with defaced version
        try:
            shutil.move(out_fname, in_fname)
        except IOError as e:
            print('I/O error({0}): {1}'.format(e.errno, e.strerror))
            print('Could not backup original image - exiting')
            sys.exit(1)

    # Cleanup temporary files
    print('Cleaning up')
    os.remove(tmp_nii_fname)
    os.remove(tmp_temp2ind_mat_fname)
    os.remove(tmp_indmask_fname)
    os.remove(tmp_img_fname)
    os.remove(tmp_mat_fname)


def mgz2niigz(mgz_fname, niigz_fname):

    mgz_obj = nib.load(mgz_fname)
    mgz_obj.to_filename(niigz_fname)


if __name__ == "__main__":
    main()
