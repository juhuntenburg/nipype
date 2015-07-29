from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename

import nibabel as nb
import numpy as np
import os
import intensity


class IntensityBackgroundEstimatorInputSpec(BaseInterfaceInputSpec):
    volume = File(exists=True, desc='volume for which background should be'
                  'estimated (nii or nii.gz)',
                  mandatory=True)
    distribution = traits.String('exponential', desc='either exponential or'
                                 'half-normal, default is exponential',
                                 usedefault=True)
    ratio = traits.Float(0.01, desc='ratio Parameter, default is 0.01',
                         usedefault=True)


class IntensityBackgroundEstimatorOutputSpec(TraitedSpec):
    masked_volume = File(exists=True, desc="input volume with masked"
                         "background")
    mask = File(exists=True, desc="background mask")
    prob_volume = File(exists=True, desx="background probability volume")


class IntensityBackgroundEstimator(BaseInterface):
    input_spec = IntensityBackgroundEstimatorInputSpec
    output_spec = IntensityBackgroundEstimatorOutputSpec

    def _run_interface(self, runtime):

        fname = self.inputs.volume
        dist = self.inputs.distribution
        ratio = self.inputs.ratio
        img = nb.load(fname)
        data = img.get_data()
        affine = img.get_affine()
        header = img.get_header()  # should the header be reused as well?

        # run the estimator in virtual machine
        intensity.initVM()
        estimator = intensity.IntensityBackgroundEstimator()
        estimator.setDims(data.shape[0], data.shape[1], data.shape[2])
        estimator.setInputImage(intensity.JArray('float')((data.flatten()).astype(float)))
        estimator.setDistributionType(dist)
        estimator.setRobustnessRatio(ratio)
        estimator.execute()

        # recast and save output data
        masked_data = np.reshape(np.array(estimator.getMaskedImage(),
                                 dtype=np.float32), data.shape)
        mask_data = np.reshape(np.array(estimator.getMask(),
                               dtype=np.uint8), data.shape)
        prob_data = np.reshape(np.array(estimator.getProbaImage(),
                               dtype=np.float32), data.shape)

        masked_img = nb.Nifti1Image(masked_data, affine, header)
        mask_img = nb.Nifti1Image(mask_data, affine, header)
        prob_img = nb.Nifti1Image(prob_data, affine, header)

        _, base, _ = split_filename(fname)
        nb.save(masked_img, base + '_masked.nii.gz')
        nb.save(mask_img, base + '_mask.nii.gz')
        nb.save(prob_img, base + '_prob.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume
        _, base, _ = split_filename(fname)
        outputs["masked_volume"] = os.path.abspath(base + '_masked.nii.gz')
        outputs["mask"] = os.path.abspath(base + '_mask.nii.gz')
        outputs["prob_volume"] = os.path.abspath(base + '_prob.nii.gz')
        return outputs
