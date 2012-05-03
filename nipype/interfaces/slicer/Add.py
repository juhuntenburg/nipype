from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class AddInputSpec(CommandLineInputSpec):
    inputVolume1 = File(position="0", desc="Input volume 1", exists=True, argstr="--inputVolume1 %s")
    inputVolume2 = File(position="1", desc="Input volume 2", exists=True, argstr="--inputVolume2 %s")
    outputVolume = traits.Either(traits.Bool, File(), position="2", hash_files=False, desc="Volume1 + Volume2", argstr="--outputVolume %s")
    order = traits.Enum("0", "1", "2", "3", desc="Interpolation order if two images are in different coordinate frames or have different sampling.", argstr="--order %s")


class AddOutputSpec(TraitedSpec):
    outputVolume = File(position="2", desc="Volume1 + Volume2", exists=True)


class Add(SlicerCommandLine):
    """title: Add Images

category: Filtering.Arithmetic

description: 
Adds two images. Although all image types are supported on input, only signed types are produced. The two images do not have to have the same dimensions.


version: 0.1.0.$Revision: 18864 $(alpha)

documentation-url: http://slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/Add

contributor: Bill Lorensen

acknowledgements: 
This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.


"""

    input_spec = AddInputSpec
    output_spec = AddOutputSpec
    _cmd = " Add "
    _outputs_filenames = {'outputVolume':'outputVolume.nii'}
