from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class DicomToNrrdConverterInputSpec(CommandLineInputSpec):
    inputDicomDirectory = Directory(desc="Directory holding Dicom series", exists=True, argstr="--inputDicomDirectory %s")
    outputDirectory = traits.Either(traits.Bool, Directory(), hash_files=False, desc="Directory holding the output NRRD format", argstr="--outputDirectory %s")
    outputVolume = traits.Str(desc="Output filename (.nhdr or .nrrd)", argstr="--outputVolume %s")
    smallGradientThreshold = traits.Float(desc="If a gradient magnitude is greater than 0 and less than smallGradientThreshold, then DicomToNrrdConverter will display an error message and quit, unless the useBMatrixGradientDirections option is set.", argstr="--smallGradientThreshold %f")
    writeProtocolGradientsFile = traits.Bool(desc=" Write the protocol gradients to a file suffixed by \".txt\" as they were specified in the procol by multiplying each diffusion gradient direction by the measurement frame.  This file is for debugging purposes only, the format is not fixed, and will likely change as debugging of new dicom formats is necessary. ", argstr="--writeProtocolGradientsFile ")
    useIdentityMeaseurementFrame = traits.Bool(desc="Adjust all the gradients so that the measurement frame is an identity matrix.", argstr="--useIdentityMeaseurementFrame ")
    useBMatrixGradientDirections = traits.Bool(desc="Fill the nhdr header with the gradient directions and bvalues computed out of the BMatrix. Only changes behavior for Siemens data.", argstr="--useBMatrixGradientDirections ")


class DicomToNrrdConverterOutputSpec(TraitedSpec):
    outputDirectory = Directory(desc="Directory holding the output NRRD format", exists=True)


class DicomToNrrdConverter(SlicerCommandLine):
    """title: 
  Dicom to Nrrd Converter 
  

category: 
  Converters
  

description: 
Converts diffusion weighted MR images in dicom series into Nrrd format for analysis in Slicer. This program has been tested on only a limited subset of DTI dicom formats available from Siemens, GE, and Phillips scanners. Work in progress to support dicom multi-frame data. The program parses dicom header to extract necessary information about measurement frame, diffusion weighting directions, b-values, etc, and write out a nrrd image. For non-diffusion weighted dicom images, it loads in an entire dicom series and writes out a single dicom volume in a .nhdr/.raw pair.
  

version: 0.2.0.$Revision: 916 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/DicomToNrrdConverter

license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt 

contributor: Xiaodong Tao

acknowledgements: 
This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.  Additional support for DTI data produced on Philips scanners was contributed by Vincent Magnotta and Hans Johnson at the University of Iowa.


"""

    input_spec = DicomToNrrdConverterInputSpec
    output_spec = DicomToNrrdConverterOutputSpec
    _cmd = " DicomToNrrdConverter "
    _outputs_filenames = {'outputDirectory':'outputDirectory'}
