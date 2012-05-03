from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os
from nipype.interfaces.slicer.base import SlicerCommandLine


class CurvatureAnisotropicDiffusionInputSpec(CommandLineInputSpec):
    conductance = traits.Float(desc="Conductance controls the sensitivity of the conductance term. As a general rule, the lower the value, the more strongly the filter preserves edges. A high value will cause diffusion (smoothing) across edges. Note that the number of iterations controls how much smoothing is done within regions bounded by edges.", argstr="--conductance %f")
    iterations = traits.Int(desc="The more iterations, the more smoothing. Each iteration takes the same amount of time. If it takes 10 seconds for one iteration, then it will take 100 seconds for 10 iterations. Note that the conductance controls how much each iteration smooths across edges.", argstr="--iterations %d")
    timeStep = traits.Float(desc="The time step depends on the dimensionality of the image. In Slicer the images are 3D and the default (.0625) time step will provide a stable solution.", argstr="--timeStep %f")
    inputVolume = File(position="0", desc="Input volume to be filtered", exists=True, argstr="--inputVolume %s")
    outputVolume = traits.Either(traits.Bool, File(), position="1", hash_files=False, desc="Output filtered", argstr="--outputVolume %s")


class CurvatureAnisotropicDiffusionOutputSpec(TraitedSpec):
    outputVolume = File(position="1", desc="Output filtered", exists=True)


class CurvatureAnisotropicDiffusion(SlicerCommandLine):
    """title: Curvature Anisotropic Diffusion

category: Filtering.Denoising

description: 
Performs anisotropic diffusion on an image using a modified curvature diffusion equation (MCDE).

MCDE does not exhibit the edge enhancing properties of classic anisotropic diffusion, which can under certain conditions undergo a 'negative' diffusion, which enhances the contrast of edges.  Equations of the form of MCDE always undergo positive diffusion, with the conductance term only varying the strength of that diffusion. 

 Qualitatively, MCDE compares well with other non-linear diffusion techniques.  It is less sensitive to contrast than classic Perona-Malik style diffusion, and preserves finer detailed structures in images.  There is a potential speed trade-off for using this function in place of Gradient Anisotropic Diffusion.  Each iteration of the solution takes roughly twice as long.  Fewer iterations, however, may be required to reach an acceptable solution.


version: 0.1.0.$Revision: 18864 $(alpha)

documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.0/Modules/CurvatureAnisotropicDiffusion

contributor: Bill Lorensen

acknowledgements: This command module was derived from Insight/Examples (copyright) Insight Software Consortium

"""

    input_spec = CurvatureAnisotropicDiffusionInputSpec
    output_spec = CurvatureAnisotropicDiffusionOutputSpec
    _cmd = " CurvatureAnisotropicDiffusion "
    _outputs_filenames = {'outputVolume':'outputVolume.nii'}
