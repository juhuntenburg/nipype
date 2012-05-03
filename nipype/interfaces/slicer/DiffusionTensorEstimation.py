from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class DiffusionTensorEstimationInputSpec(CommandLineInputSpec):
    inputVolume = File(position="0", desc="Input DWI volume", exists=True, argstr="--inputVolume %s")
    mask = File(desc="Mask where the tensors will be computed", exists=True, argstr="--mask %s")
    outputTensor = traits.Either(traits.Bool, File(), position="1", hash_files=False, desc="Estimated DTI volume", argstr="--outputTensor %s")
    outputBaseline = traits.Either(traits.Bool, File(), position="2", hash_files=False, desc="Estimated baseline volume", argstr="--outputBaseline %s")
    enumeration = traits.Enum("LS", "WLS", desc="LS: Least Squares, WLS: Weighted Least Squares", argstr="--enumeration %s")
    shiftNeg = traits.Bool(desc="Shift eigenvalues so all are positive (accounts for bad tensors related to noise or acquisition error)", argstr="--shiftNeg ")


class DiffusionTensorEstimationOutputSpec(TraitedSpec):
    outputTensor = File(position="1", desc="Estimated DTI volume", exists=True)
    outputBaseline = File(position="2", desc="Estimated baseline volume", exists=True)


class DiffusionTensorEstimation(SlicerCommandLine):
    """title: 
  Diffusion Tensor Estimation
  

category: 
  Diffusion.Utilities
  

description: 
  Performs a tensor model estimation from diffusion weighted images. 

There are three estimation methods available: least squares, weigthed least squares and non-linear estimation. The first method is the traditional method for tensor estimation and the fastest one. Weighted least squares takes into account the noise characteristics of the MRI images to weight the DWI samples used in the estimation based on its intensity magnitude. The last method is the more complex.
  

version: 0.1.0.$Revision: 1892 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/DiffusionTensorEstimation

license: slicer3

contributor: Raul San Jose

acknowledgements: This command module is based on the estimation functionality provided by the Teem library. This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149. 

"""

    input_spec = DiffusionTensorEstimationInputSpec
    output_spec = DiffusionTensorEstimationOutputSpec
    _cmd = " DiffusionTensorEstimation "
    _outputs_filenames = {'outputTensor':'outputTensor.nii','outputBaseline':'outputBaseline.nii'}
