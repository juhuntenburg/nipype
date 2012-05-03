from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class CheckerBoardInputSpec(CommandLineInputSpec):
    checkerPattern = InputMultiPath(traits.Int, desc="The pattern of input 1 and input 2 in the output image. The user can specify the number of checkers in each dimension. A checkerPattern of 2,2,1 means that images will alternate in every other checker in the first two dimensions. The same pattern will be used in the 3rd dimension.", sep=",", argstr="--checkerPattern %s")
    inputVolume1 = File(position="0", desc="First Input volume", exists=True, argstr="--inputVolume1 %s")
    inputVolume2 = File(position="1", desc="Second Input volume", exists=True, argstr="--inputVolume2 %s")
    outputVolume = traits.Either(traits.Bool, File(), position="2", hash_files=False, desc="Output filtered", argstr="--outputVolume %s")


class CheckerBoardOutputSpec(TraitedSpec):
    outputVolume = File(position="2", desc="Output filtered", exists=True)


class CheckerBoard(SlicerCommandLine):
    """title: 
  CheckerBoard Filter
  

category: 
  Filtering
  

description: 
Create a checkerboard volume of two volumes. The output volume will show the two inputs alternating according to the user supplied checkerPattern. This filter is often used to compare the results of image registration. Note that the second input is resampled to the same origin, spacing and direction before it is composed with the first input. The scalar type of the output volume will be the same as the input image scalar type.
  

version: 0.1.0.$Revision: 18864 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/CheckerBoard

contributor: Bill Lorensen

acknowledgements: 
This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.


"""

    input_spec = CheckerBoardInputSpec
    output_spec = CheckerBoardOutputSpec
    _cmd = " CheckerBoard "
    _outputs_filenames = {'outputVolume':'outputVolume.nii'}
