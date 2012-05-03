from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class BSplineDeformableRegistrationInputSpec(CommandLineInputSpec):
    iterations = traits.Int(desc="Number of iterations", argstr="--iterations %d")
    gridSize = traits.Int(desc="Number of grid points on interior of the fixed image. Larger grid sizes allow for finer registrations.", argstr="--gridSize %d")
    histogrambins = traits.Int(desc="Number of histogram bins to use for Mattes Mutual Information. Reduce the number of bins if a deformable registration fails. If the number of bins is too large, the estimated PDFs will be a field of impulses and will inhibit reliable registration estimation.", argstr="--histogrambins %d")
    spatialsamples = traits.Int(desc="Number of spatial samples to use in estimating Mattes Mutual Information. Larger values yield more accurate PDFs and improved registration quality.", argstr="--spatialsamples %d")
    constrain = traits.Bool(desc="Constrain the deformation to the amount specified in Maximum Deformation", argstr="--constrain ")
    maximumDeformation = traits.Float(desc="If Constrain Deformation is checked, limit the deformation to this amount.", argstr="--maximumDeformation %f")
    default = traits.Int(desc="Default pixel value used if resampling a pixel outside of the volume.", argstr="--default %d")
    initialtransform = File(desc="Initial transform for aligning the fixed and moving image. Maps positions in the fixed coordinate frame to positions in the moving coordinate frame. This transform should be an affine or rigid transform.  It is used an a bulk transform for the BSpline. Optional.", exists=True, argstr="--initialtransform %s")
    FixedImageFileName = File(position="0", desc="Fixed image to which to register", exists=True, argstr="--FixedImageFileName %s")
    MovingImageFileName = File(position="1", desc="Moving image", exists=True, argstr="--MovingImageFileName %s")
    outputtransform = traits.Either(traits.Bool, File(), hash_files=False, desc="Transform calculated that aligns the fixed and moving image. Maps positions from the fixed coordinate frame to the moving coordinate frame. Optional (specify an output transform or an output volume or both).", argstr="--outputtransform %s")
    outputwarp = traits.Either(traits.Bool, File(), hash_files=False, desc="Vector field that applies an equivalent warp as the BSpline. Maps positions from the fixed coordinate frame to the moving coordinate frame. Optional.", argstr="--outputwarp %s")
    resampledmovingfilename = traits.Either(traits.Bool, File(), hash_files=False, desc="Resampled moving image to fixed image coordinate frame. Optional (specify an output transform or an output volume or both).", argstr="--resampledmovingfilename %s")


class BSplineDeformableRegistrationOutputSpec(TraitedSpec):
    outputtransform = File(desc="Transform calculated that aligns the fixed and moving image. Maps positions from the fixed coordinate frame to the moving coordinate frame. Optional (specify an output transform or an output volume or both).", exists=True)
    outputwarp = File(desc="Vector field that applies an equivalent warp as the BSpline. Maps positions from the fixed coordinate frame to the moving coordinate frame. Optional.", exists=True)
    resampledmovingfilename = File(desc="Resampled moving image to fixed image coordinate frame. Optional (specify an output transform or an output volume or both).", exists=True)


class BSplineDeformableRegistration(SlicerCommandLine):
    """title: Fast Nonrigid BSpline registration

category: Legacy.Registration

description: Registers two images together using BSpline transform and mutual information.

version: 0.1.0.$Revision: 18864 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/BSplineDeformableRegistration

contributor: Bill Lorensen

acknowledgements: 
This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.


"""

    input_spec = BSplineDeformableRegistrationInputSpec
    output_spec = BSplineDeformableRegistrationOutputSpec
    _cmd = " BSplineDeformableRegistration "
    _outputs_filenames = {'resampledmovingfilename':'resampledmovingfilename.nii','outputtransform':'outputtransform.txt','outputwarp':'outputwarp.nrrd'}
