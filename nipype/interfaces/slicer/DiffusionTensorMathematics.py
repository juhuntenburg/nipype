from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class DiffusionTensorMathematicsInputSpec(CommandLineInputSpec):
    inputVolume = File(position="0", desc="Input DTI volume", exists=True, argstr="--inputVolume %s")
    outputScalar = traits.Either(traits.Bool, File(), position="2", hash_files=False, desc="Scalar volume derived from tensor", argstr="--outputScalar %s")
    enumeration = traits.Enum("Trace", "Determinant", "RelativeAnisotropy", "FractionalAnisotropy", "Mode", "LinearMeasure", "PlanarMeasure", "SphericalMeasure", "MinEigenvalue", "MidEigenvalue", "MaxEigenvalue", "MaxEigenvalueProjectionX", "MaxEigenvalueProjectionY", "MaxEigenvalueProjectionZ", "RAIMaxEigenvecX", "RAIMaxEigenvecY", "RAIMaxEigenvecZ", "D11", "D22", "D33", "ParallelDiffusivity", "PerpendicularDffusivity", desc="An enumeration of strings", argstr="--enumeration %s")


class DiffusionTensorMathematicsOutputSpec(TraitedSpec):
    outputScalar = File(position="2", desc="Scalar volume derived from tensor", exists=True)


class DiffusionTensorMathematics(SlicerCommandLine):
    """title: 
  Diffusion Tensor Scalar Measurements
  

category: 
  Diffusion.Utilities
  

description: 
  Compute a set of different scalar measurements from a tensor field, specially oriented for Diffusion Tensors where some rotationally invariant measurements, like Fractional Anisotropy, are highly used to describe the anistropic behaviour of the tensor.
  

version: 0.1.0.$Revision: 1892 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/DiffusionTensorMathematics

contributor: Raul San Jose

acknowledgements: LMI

"""

    input_spec = DiffusionTensorMathematicsInputSpec
    output_spec = DiffusionTensorMathematicsOutputSpec
    _cmd = " DiffusionTensorMathematics "
    _outputs_filenames = {'outputScalar':'outputScalar.nii'}
