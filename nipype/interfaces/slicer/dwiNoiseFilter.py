from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class dwiNoiseFilterInputSpec(CommandLineInputSpec):
    iter = traits.Int(desc="Number of iterations for the noise removal filter.", argstr="--iter %d")
    re = InputMultiPath(traits.Int, desc="Estimation radius.", sep=",", argstr="--re %s")
    rf = InputMultiPath(traits.Int, desc="Filtering radius.", sep=",", argstr="--rf %s")
    mnvf = traits.Int(desc="Minimum number of voxels in kernel used for filtering.", argstr="--mnvf %d")
    mnve = traits.Int(desc="Minimum number of voxels in kernel used for estimation.", argstr="--mnve %d")
    minnstd = traits.Int(desc="Minimum allowed noise standard deviation.", argstr="--minnstd %d")
    maxnstd = traits.Int(desc="Maximum allowed noise standard deviation.", argstr="--maxnstd %d")
    hrf = traits.Float(desc="How many histogram bins per unit interval.", argstr="--hrf %f")
    uav = traits.Bool(desc="Use absolute value in case of negative square.", argstr="--uav ")
    inputVolume = File(position="0", desc="Input DWI volume.", exists=True, argstr="--inputVolume %s")
    outputVolume = traits.Either(traits.Bool, File(), position="1", hash_files=False, desc="Output DWI volume.", argstr="--outputVolume %s")


class dwiNoiseFilterOutputSpec(TraitedSpec):
    outputVolume = File(position="1", desc="Output DWI volume.", exists=True)


class dwiNoiseFilter(SlicerCommandLine):
    """title: Rician LMMSE Image Filter

category: Diffusion.Denoising

description: 
This module reduces noise (or unwanted detail) on a set of diffusion weighted images. For this, it filters the image in the mean squared error sense using a Rician noise model. Images corresponding to each gradient direction, including baseline, are processed individually. The noise parameter is automatically estimated (noise estimation improved but slower).
Note that this is a general purpose filter for MRi images. The module jointLMMSE has been specifically designed for DWI volumes and shows a better performance, so its use is recommended instead.
A complete description of the algorithm in this module can be found in:
S. Aja-Fernandez, M. Niethammer, M. Kubicki, M. Shenton, and C.-F. Westin. Restoration of DWI data using a Rician LMMSE estimator. IEEE Transactions on Medical Imaging, 27(10): pp. 1389-1403, Oct. 2008. 


version: 0.1.1.$Revision: 1 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/RicianLMMSEImageFilter

contributor: Antonio Tristan Vega, Santiago Aja Fernandez and Marc Niethammer. Partially founded by grant number TEC2007-67073/TCM from the Comision Interministerial de Ciencia y Tecnologia (Spain).

"""

    input_spec = dwiNoiseFilterInputSpec
    output_spec = dwiNoiseFilterOutputSpec
    _cmd = " dwiNoiseFilter "
    _outputs_filenames = {'outputVolume':'outputVolume.nii'}
