from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class AffineRegistrationInputSpec(CommandLineInputSpec):
    fixedsmoothingfactor = traits.Int(desc="Amount of smoothing applied to fixed image prior to registration. Default is 0 (none). Range is 0-5 (unitless). Consider smoothing the input data if there is considerable amounts of noise or the noise pattern in the fixed and moving images is very different.", argstr="--fixedsmoothingfactor %d")
    movingsmoothingfactor = traits.Int(desc="Amount of smoothing applied to moving image prior to registration. Default is 0 (none). Range is 0-5 (unitless). Consider smoothing the input data if there is considerable amounts of noise or the noise pattern in the fixed and moving images is very different.", argstr="--movingsmoothingfactor %d")
    histogrambins = traits.Int(desc="Number of histogram bins to use for Mattes Mutual Information. Reduce the number of bins if a registration fails. If the number of bins is too large, the estimated PDFs will be a field of impulses and will inhibit reliable registration estimation.", argstr="--histogrambins %d")
    spatialsamples = traits.Int(desc="Number of spatial samples to use in estimating Mattes Mutual Information. Larger values yield more accurate PDFs and improved registration quality.", argstr="--spatialsamples %d")
    iterations = traits.Int(desc="Number of iterations", argstr="--iterations %d")
    translationscale = traits.Float(desc="Relative scale of translations to rotations, i.e. a value of 100 means 10mm = 1 degree. (Actual scale used is 1/(TranslationScale^2)). This parameter is used to \"weight\" or \"standardized\" the transform parameters and their effect on the registration objective function.", argstr="--translationscale %f")
    initialtransform = File(desc="Initial transform for aligning the fixed and moving image.  Maps positions in the fixed coordinate frame to positions in the moving coordinate frame. Optional.", exists=True, argstr="--initialtransform %s")
    FixedImageFileName = File(position="0", desc="Fixed image to which to register", exists=True, argstr="--FixedImageFileName %s")
    MovingImageFileName = File(position="1", desc="Moving image", exists=True, argstr="--MovingImageFileName %s")
    outputtransform = traits.Either(traits.Bool, File(), hash_files=False, desc="Transform calculated that aligns the fixed and moving image. Maps positions in the fixed coordinate frame to the moving coordinate frame. Optional (specify an output transform or an output volume or both).", argstr="--outputtransform %s")
    resampledmovingfilename = traits.Either(traits.Bool, File(), hash_files=False, desc="Resampled moving image to the fixed image coordinate frame. Optional (specify an output transform or an output volume or both).", argstr="--resampledmovingfilename %s")


class AffineRegistrationOutputSpec(TraitedSpec):
    outputtransform = File(desc="Transform calculated that aligns the fixed and moving image. Maps positions in the fixed coordinate frame to the moving coordinate frame. Optional (specify an output transform or an output volume or both).", exists=True)
    resampledmovingfilename = File(desc="Resampled moving image to the fixed image coordinate frame. Optional (specify an output transform or an output volume or both).", exists=True)


class AffineRegistration(SlicerCommandLine):
    """title: Fast Affine registration

category: Legacy.Registration

description: Registers two images together using an affine transform and mutual information. This module is often used to align images of different subjects or images of the same subject from different modalities.

This module can smooth images prior to registration to mitigate noise and improve convergence. Many of the registration parameters require a working knowledge of the algorithm although the default parameters are sufficient for many registration tasks.



version: 0.1.0.$Revision: 18864 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/AffineRegistration

contributor: Daniel Blezek

acknowledgements: 
This module was developed by Daniel Blezek while at GE Research with contributions from Jim Miller.

This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.


"""

    input_spec = AffineRegistrationInputSpec
    output_spec = AffineRegistrationOutputSpec
    _cmd = " AffineRegistration "
    _outputs_filenames = {'resampledmovingfilename':'resampledmovingfilename.nii','outputtransform':'outputtransform.txt'}
