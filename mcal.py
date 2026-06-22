import ngmix
import galsim
import numpy as np

class MetacalRunner:
    def __init__(self, obs_image, psf_image):
        self.obs_image = obs_image
        self.psf_image = psf_image
        return None

    def measure_shear(self, method):
        self.results = {}
        if method == "estimateShear":
            mcal_results = self.measure_shear_estimateShear(self.obs_image)
        elif method == "ngmix":
            mcal_results = self.measure_shear_ngmix(self.obs_image)
        elif method == "admomBootstrap":
            raise NotImplementedError("admomBootstrap is not implemented")
        else:
            raise ValueError(f"Unknown metacal method: {method}")

        self.results["mcal_results"] = mcal_results
        self.results["R"] = self.calculate_R(mcal_results)
        self.results["g_uncal"] = mcal_results['noshear']['g']
        return 0
    
    def extract_branch_g(self, gal_result, branch):
        return np.array(gal_result[branch]["g"])

    def calculate_R(self, gal_result, step=0.01):
        g_1p = self.extract_branch_g(gal_result, "1p")
        g_1m = self.extract_branch_g(gal_result, "1m")
        g_2p = self.extract_branch_g(gal_result, "2p")
        g_2m = self.extract_branch_g(gal_result, "2m")
        delta_gamma = 2*step
        R11 = (g_1p[0] - g_1m[0]) / delta_gamma
        R22 = (g_2p[1] - g_2m[1]) / delta_gamma
        R12 = (g_1p[1] - g_1m[1]) / delta_gamma
        R21 = (g_2p[0] - g_2m[0]) / delta_gamma

        return np.array([[R11, R12], [R21, R22]])

    def measure_shear_ngmix(self, obs_image):
        psf_obs = ngmix.observation.Observation(self.psf_image.array)
        pfitter = ngmix.fitting.Fitter("gauss")
        pfit_guess = self.make_guess(self.psf_image.array)

        pfit_result = pfitter.go(psf_obs, pfit_guess)
        psf_gmix_fit = pfit_result.get_gmix()

        psf_obs.set_gmix(psf_gmix_fit)

        weight = np.ones(obs_image.array.shape)
        obs = ngmix.observation.Observation(obs_image.array, weight=weight, psf=psf_obs)

        obdic = ngmix.metacal.get_all_metacal(obs, fixnoise=False)

        mcal_results = {}

        for key in obdic:
            mobs = obdic[key]
            this_psf = mobs.get_psf()
            this_image = mobs.image

            this_pfitter = ngmix.fitting.Fitter("gauss")
            this_pguess = self.make_guess(this_psf.image)
            this_pfit_result = this_pfitter.go(this_psf, this_pguess)
            this_psf_gmix_fit = this_pfit_result.get_gmix()

            this_psf.set_gmix(this_psf_gmix_fit)

            weight = np.ones(this_image.shape)
            this_obs = ngmix.observation.Observation(this_image, weight=weight, psf=this_psf)

            this_fitter = ngmix.fitting.Fitter("gauss")
            this_guess = self.make_guess(this_image)
            this_res = this_fitter.go(this_obs, this_guess)
            mcal_results[key] = {
                "g": np.array(this_res["g"]),
                "flags": this_res["flags"],
            }

        return mcal_results
    
    def measure_shear_estimateShear(self, obs_image):
        psf_obs = ngmix.Observation(self.psf_image.array)
        obs = ngmix.Observation(obs_image.array, psf=psf_obs)

        obdic = ngmix.metacal.get_all_metacal(obs, fixnoise=False)

        mcal_results = {}

        for key in obdic:
            mobs = obdic[key]
            mpsf_array = mobs.get_psf().image
            mimage_array = mobs.image

            this_image = galsim.Image(mimage_array)
            this_image_epsf = galsim.Image(mpsf_array)

            res = galsim.hsm.EstimateShear(this_image, this_image_epsf)

            res_shear = galsim.Shear(e1=res.corrected_e1, e2=res.corrected_e2)
            mcal_results[key] = {"g": np.array([res_shear.g1, res_shear.g2])}

        return mcal_results

    def make_guess(self, array):
        eps = 0.01
        # shape = galsim.hsm.FindAdaptiveMom(galsim.Image(array))
        pars = np.zeros(6)
        pars[0] = 0
        pars[1] = 0
        pars[2] = 0
        pars[3] = 0
        pars[4] = 100
        pars[5] = 1
        return pars

    def get_results(self):
        return self.results