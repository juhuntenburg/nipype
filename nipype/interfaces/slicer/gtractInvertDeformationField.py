from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class gtractInvertDeformationFieldInputSpec(CommandLineInputSpec):
    baseImage = File(desc="Required: base image used to define the size of the inverse field", exists=True, argstr="--baseImage %s")
    deformationImage = File(desc="Required: Deformation field image", exists=True, argstr="--deformationImage %s")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Required: Output deformation field", argstr="--outputVolume %s")
    subsamplingFactor = traits.Int(desc="Subsampling factor for the deformation field", argstr="--subsamplingFactor %d")
    numberOfThreads = traits.Int(desc="Explicitly specify the maximum number of threads to use.", argstr="--numberOfThreads %d")


class gtractInvertDeformationFieldOutputSpec(TraitedSpec):
    outputVolume = File(desc="Required: Output deformation field", exists=True)


class gtractInvertDeformationField(SlicerCommandLine):
    """title: Invert Deformation Field

category: Diffusion.GTRACT

description: This program will invert a deformatrion field. The size of the deformation field is defined by an example image provided by the user

version: 4.0.0

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

contributor: This tool was developed by Vincent Magnotta.

acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1

"""

    input_spec = gtractInvertDeformationFieldInputSpec
    output_spec = gtractInvertDeformationFieldOutputSpec
    _cmd = " gtractInvertDeformationField "
    _outputs_filenames = {'outputVolume':'outputVolume.nrrd'}
