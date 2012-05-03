from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class DiffusionWeightedMaskingInputSpec(CommandLineInputSpec):
    inputVolume = File(position="0", desc="Input DWI volume", exists=True, argstr="--inputVolume %s")
    outputBaseline = traits.Either(traits.Bool, File(), position="2", hash_files=False, desc="Estimated baseline volume", argstr="--outputBaseline %s")
    thresholdMask = traits.Either(traits.Bool, File(), position="3", hash_files=False, desc="Otsu Threshold Mask", argstr="--thresholdMask %s")
    otsuomegathreshold = traits.Float(desc="Control the sharpness of the threshold in the Otsu computation. 0: lower threshold, 1: higher threhold", argstr="--otsuomegathreshold %f")
    removeislands = traits.Bool(desc="Remove Islands in Threshold Mask?", argstr="--removeislands ")


class DiffusionWeightedMaskingOutputSpec(TraitedSpec):
    outputBaseline = File(position="2", desc="Estimated baseline volume", exists=True)
    thresholdMask = File(position="3", desc="Otsu Threshold Mask", exists=True)


class DiffusionWeightedMasking(SlicerCommandLine):
    """title: 
  Mask from Diffusion Weighted Images
  

category: 
  Diffusion.Utilities
  

description: <p>Performs a mask calculation from a diffusion weighted (DW) image.</p><p>Starting from a dw image, this module computes the baseline image averaging all the images without diffusion weighting and then applies the otsu segmentation algorithm in order to produce a mask. this mask can then be used when estimating the diffusion tensor (dt) image, not to estimate tensors all over the volume.</p>

version: 0.1.0.$Revision: 1892 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/DiffusionWeightedMasking

license: slicer3

contributor: Demian Wassermann

"""

    input_spec = DiffusionWeightedMaskingInputSpec
    output_spec = DiffusionWeightedMaskingOutputSpec
    _cmd = " DiffusionWeightedMasking "
    _outputs_filenames = {'outputBaseline':'outputBaseline.nii','thresholdMask':'thresholdMask.nii'}
