#!/usr/bin/env python3
"""
Russ Poldrack's pydeface modified to perform non-invertable face voxelation
- Retains deidentified signal in face area for more stable registration
- Optional export of registered face mask
- Optional import and application of previously calculated face mask

AUTHOR : Mike Tyszka, Ph.D.
PLACE  : Caltech Brain Imaging Center

Original Copyright:

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
import shutil
import nibabel as nib
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
    T1w_template = resource_filename(Requirement.parse("pydeface"), "pydeface/data/ConteCore2_50_T1w_2mm.nii.gz")
    facemask = resource_filename(Requirement.parse("pydeface"), "pydeface/data/ConteCore2_50_T1w_2mm_deface_mask.nii.gz")

    try:
        assert os.path.exists(T1w_template)
    except:
        raise Exception('*** Missing template {}'.format(T1w_template))

    try:
        assert os.path.exists(facemask)
    except:
        raise Exception('*** Missing facemask {}'.format(facemask))

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
    parser.add_argument('-im', '--inmask', required= False,
                        help='Use this Nifti face mask to perform defacing')
    parser.add_argument('-om', '--outmask', required= False,
                        help='Save registered face mask to this Nifti file')
    parser.add_argument('-r', '--replace', required=False, default=False, action='store_true',
                        help='Backup original image and replace with defaced version')
    parser.add_argument('--overwrite', required=False, default=False, action='store_true',
                        help='Overwrite existing defaced output')

    # Parse command line arguments
    args = parser.parse_args()
    in_fname = args.infile

    print('Input faced image    : {}'.format(in_fname))

    # Save full image extension
    # Handle double-extension for compressed Nifti files
    if not in_fname.endswith('.nii.gz'):
        print('* pydeface currently only supports .nii.gz images - exiting')
        sys.exit(1)

    # Output defaced image filename
    if args.outfile:
        out_fname = args.outfile
    else:
        out_fname = in_fname.replace('.nii.gz', '_defaced.nii.gz')

    # Protect existing output file
    if os.path.isfile(out_fname) and not args.overwrite:
        print('{} already exists - remove it first'.format(out_fname))
        sys.exit(1)

    print('Output defaced image : {}'.format(out_fname))

    if args.scalefactor:
        vox_sf = float(args.scalefactor)
    else:
        vox_sf = 8.0

    print('Defacing {}'.format(in_fname))

    # Load input image
    print('Loading {}'.format(in_fname))
    in_nii = nib.load(in_fname)
    in_img = in_nii.get_fdata()

    # Voxelate input image
    # 1. Cubic downsample
    # 2. Nearest neighbor upsample
    print('Voxelating faced image : scale factor %0.1f' % vox_sf)

    print('  Spline downsampling')
    in_vox_img = nd.interpolation.zoom(in_img, zoom=1.0/vox_sf, order=3)

    print('  Nearest neighbor upsampling')
    in_vox_img = nd.interpolation.zoom(in_vox_img, zoom=vox_sf, order=0)

    if args.inmask:

        # Load precalculated individual space face mask
        ind_deface_mask_nii = nib.load(args.inmask)
        ind_deface_mask_img = ind_deface_mask_nii.get_fdata()

    else:

        # Create temporary directory for FLIRT outputs
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_obj.name
        flirt_tx = os.path.join(tmp_dir, 'flirt_tx.mat')
        dummy_mat = os.path.join(tmp_dir, 'dummy.mat')
        dummy_img = os.path.join(tmp_dir, 'dummy.nii.gz')
        deface_mask_fname = os.path.join(tmp_dir, 'deface_mask.nii.gz')

        # Register template to infile
        print('Registering template to individual space')
        flirt = fsl.FLIRT()
        flirt.inputs.cost_func='mutualinfo'
        flirt.inputs.in_file = T1w_template
        flirt.inputs.out_matrix_file = flirt_tx
        flirt.inputs.reference = in_fname
        flirt.inputs.out_file = dummy_img
        flirt.terminal_output='none'
        flirt.run()

        # Affine transform facemask to infile
        print('Resampling face mask to individual space')
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = facemask
        flirt.inputs.in_matrix_file = flirt_tx
        flirt.inputs.apply_xfm = True
        flirt.inputs.reference = in_fname
        flirt.inputs.out_file = deface_mask_fname
        flirt.inputs.out_matrix_file = dummy_mat
        flirt.terminal_output='none'
        flirt.run()

        # Load computed individual space deface mask
        ind_deface_mask_nii = nib.load(deface_mask_fname)
        ind_deface_mask_img = ind_deface_mask_nii.get_fdata()

    # Replace face area with voxelated version
    # Note that the face mask is 0 in the face region, 1 elsewhere.
    print('Anonymizing face area')
    out_img = in_img * ind_deface_mask_img + in_vox_img * (1 - ind_deface_mask_img)

    # Save defaced image
    print('Saving defaced image to {}'.format(out_fname))
    defaced_nii = nib.Nifti1Image(out_img, in_nii.affine, in_nii.header)
    defaced_nii.to_filename(out_fname)

    # Backup and replace original if requested
    if args.replace:

        bak_fname = in_fname.replace('.nii.gz', '_bak.nii.gz')

        print('Backup up {} to {}'.format(in_fname, bak_fname))
        shutil.move(in_fname, bak_fname)

        print('Renaming {} to {}'.format(out_fname, in_fname))
        shutil.move(out_fname, in_fname)

    # Save mask if requested
    if args.outmask:
        print('Saving registered face mask to {}'.format(args.outmask))
        ind_deface_mask_nii.to_filename(args.outmask)


if __name__ == "__main__":
    main()
