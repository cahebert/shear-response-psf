"""
Shapelet-based PSF modification and shear measurement using metacalibration.

Code originally written by TQ Zhang, modifified by CAH.
"""
import numpy as np
import galsim
import scipy.linalg as alg
import mcal
import momhelper

class shapeletXmoment:
    def __init__(self, psf,n, bmax=10, pixel_scale=1.0):
        self.n = n
        self.bmax = bmax
        self.pixel_scale = pixel_scale
        self.base_psf = psf
        self.base_psf_image = psf.drawImage(scale = pixel_scale)
        self.base_psf_result = galsim.hsm.FindAdaptiveMom(self.base_psf_image)
        self.base_shapelet = galsim.Shapelet.fit(self.base_psf_result.moments_sigma, bmax, self.base_psf_image, normalization = 'sb')
        self.base_bvec = self.base_shapelet.bvec

    def moment_measure(self, image, p, q):
        n = p+q
        if n<2:
            print( "Does not support moment measure less than second order.")
            return 0
        elif n==2:
            return self.get_second_moment(image, p, q)
        else:
            return self.higher_weighted_moment(image, p, q)

    def get_second_moment(self, image, p, q):
        image_results = galsim.hsm.FindAdaptiveMom(image)
        if p==2:
            return image_results.observed_shape.e1
        elif q==2:
            return image_results.observed_shape.e2
        else:
            return image_results.moments_sigma

    def higher_weighted_moment(self, gsimage, p, q):
        image = gsimage.array
        y, x = np.mgrid[:image.shape[0],:image.shape[1]]+1

        psfresults = galsim.hsm.FindAdaptiveMom(galsim.Image(image, scale=self.pixel_scale))
        M = np.zeros((2,2))
        e1 = psfresults.observed_shape.e1
        e2 = psfresults.observed_shape.e2
        sigma4 = psfresults.moments_sigma**4
        c = (1+e1)/(1-e1)
        M[1][1] = np.sqrt(sigma4/(c-0.25*e2**2*(1+c)**2))
        M[0][0] = c*M[1][1]
        M[0][1] = 0.5*e2*(M[1][1]+M[0][0])
        M[1][0] = M[0][1]

        pos = np.array([x-psfresults.moments_centroid.x, y-psfresults.moments_centroid.y])
        pos = np.swapaxes(pos,0,1)
        pos = np.swapaxes(pos,1,2)

        inv_M = np.linalg.inv(M)
        sqrt_inv_M = alg.sqrtm(inv_M)
        std_pos = np.zeros(pos.shape)
        weight = np.zeros(pos.shape[0:2])
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                this_pos = pos[i][j]
                this_standard_pos = np.matmul(sqrt_inv_M, this_pos)
                std_pos[i][j] = this_standard_pos
                weight[i][j] = np.exp(-0.5 * this_standard_pos.dot(this_standard_pos))

        std_x, std_y = std_pos[:,:,0],std_pos[:,:,1]

        return np.sum(std_x**p * std_y**q * weight * image) / np.sum(image * weight)

    def modify_pq(self, m, c, delta=0.0001):
        n = self.n
        mu = self.get_mu(n)
        pq_list = self.get_pq_full(n)
        shapelet_list = self.pq2shapelet(pq_list)

        ori_moments = self.get_all_moments(self.base_psf_image, pq_list)

        A = np.zeros(shape =(mu, mu))

        #i is the mode index
        #j is the moment index
        #measure d_moment_j / d_mode_i

        for i in range(mu):
            mode_index = shapelet_list[i]
            pert_bvec = self.base_bvec.copy()
            pert_bvec[mode_index]+=delta
            ith_pert = galsim.Shapelet(self.base_psf_result.moments_sigma, self.bmax, pert_bvec)
            pert_moment = self.get_all_moments(ith_pert.drawImage(scale = self.pixel_scale,method = 'no_pixel'), pq_list)
            for j in range(mu):
                A[i][j] = (pert_moment[j] - ori_moments[j])/delta
        self.A = A

        dm = np.zeros(mu)
        dm += m*ori_moments + c
        ds = np.linalg.solve(A.T,dm)

        true_mod_bvec = self.base_bvec.copy()
        for i in range(mu):
            true_mod_bvec[shapelet_list[i]]+=ds[i]

        self.true_mod = galsim.Shapelet(
            self.base_psf_result.moments_sigma,
            self.bmax,
            true_mod_bve
        )
        return self.true_mod

    def step_modify_pq(self,current_moments,current_dm, current_mod_bvec ,current_psf,mu,shapelet_list,delta, pq_list):
        A = np.zeros(shape =(mu, mu))
        for i in range(mu):
            mode_index = shapelet_list[i]

            pert_bvec = current_mod_bvec.copy()
            pert_bvec[mode_index]+=delta
            ith_pert = galsim.Shapelet(
                self.base_psf_result.moments_sigma,
                self.bmax,
                pert_bvec
            )
            pert_moment = self.get_all_moments(
                ith_pert.drawImage(scale=self.pixel_scale),
                pq_list,
            )
            for j in range(mu):
                A[i][j] = (pert_moment[j] - current_moments[j])/delta

        ds = np.linalg.solve(A.T,current_dm)

        for i in range(mu):
            current_mod_bvec[shapelet_list[i]]+=ds[i]
        return current_mod_bvec

    def iterative_modify_pq(self, m, c, delta=0.0001, threshold=1e-6):
        iterative_n = 10

        n = self.n
        mu = self.get_mu(n)
        pq_list = self.get_pq_full(n)
        shapelet_list = self.pq2shapelet(pq_list)
        base_shapelet_image = self.base_shapelet.drawImage(scale=self.pixel_scale)
        original_moment = self.get_all_moments(base_shapelet_image, pq_list)
        current_moment = self.get_all_moments(base_shapelet_image, pq_list)
        current_dm = np.zeros(mu)
        current_dm += m*current_moment + c

        destiny_moment = current_moment + current_dm

        current_mod_bvec = self.base_bvec.copy()
        current_psf = galsim.Shapelet(self.base_psf_result.moments_sigma,  self.bmax, current_mod_bvec)

        while (np.max(np.abs(current_dm)) > threshold):

            current_mod_bvec = self.step_modify_pq(
                current_moment,
                current_dm, 
                current_mod_bvec, 
                current_psf,
                mu,
                shapelet_list,
                delta,
                pq_list
                )
            current_psf = galsim.Shapelet(
                self.base_psf_result.moments_sigma,
                self.bmax,
                current_mod_bvec
                )
            current_moment = self.get_all_moments(
                current_psf.drawImage(scale=self.pixel_scale),
                pq_list
            )

            current_dm = destiny_moment - current_moment

        return current_psf

    def get_all_moments(self, image, pq_list):
        results_list = []
        for tup in pq_list:
            results_list.append(self.moment_measure(image, tup[0], tup[1]))
        return np.array(results_list)

    def pq2mode(self,p,q):
        if p<=q:
            return (p+q)*(p+q+1)//2 + 2*min(p,q)
        else:
            return (p+q)*(p+q+1)//2 + 2*min(p,q)+1

    def pq2shapelet(self,pq_list):
        shapelet_index = []
        for tup in pq_list:
            shapelet_index.append(self.pq2mode(tup[0], tup[1]))
        return shapelet_index

    def get_mu(self, n):
        mu = 0
        for i in range(2,n+1):
            mu+=i+1
        return mu

    def get_pq_full(self,nmax):
        pq_list = []
        for n in range(2, nmax+1):
            p = 0
            q = n
            pq_list.append((p,q))

            while p<n:
                p+=1
                q-=1
                pq_list.append((p,q))
        return pq_list

    def get_pq_except(self,nmax,p,q):
        pq_full = self.get_pq_full(nmax)
        pq_except = []
        for tup in pq_full:
            if tup != (p,q):
                pq_except.append(tup)

        return pq_except


class HOMExShapeletPair:
    def __init__(
        self,
        gal_type, gal_sigma,
        e1, e2,
        g1, g2,
        psf_type, psf_sigma,
        gal_flux=1.e2,
        pixel_scale=1.0,
        sersicn=-1,
        psf_sersicn=-1,
        subtract_intersection=True,
        is_self_defined_PSF=False,
        self_defined_PSF=None,
        self_define_PSF_model=None,
        metacal_method='estimateShear',
        bpd_params=None,
        psf_gsobj=None):

        #Define basic variables
        self.pixel_scale = pixel_scale
        self.subtract_intersection = subtract_intersection
        self.is_self_defined_PSF = is_self_defined_PSF
        self.metacal_method = metacal_method

        #Define galaxy
        self.gal_type = gal_type
        self.gal_sigma = gal_sigma # in physical units
        self.gal_flux=gal_flux
        self.e1 = e1
        self.e2 = e2
        self.g1 = g1
        self.g2 = g2
        self.cosmic_shear = galsim.Shear(g1=g1, g2=g2)
        self.g = np.array([g1, g2])
        self.e = np.array([e1, e2])
        self.sersicn=sersicn
        self.e_truth = self.e
        self.bpd_params = bpd_params

        if gal_type == 'gaussian':
            gaussian_profile = galsim.Gaussian(sigma=gal_sigma)
            self.gal_light = gaussian_profile.withFlux(self.gal_flux)
            self.gal_light = self.gal_light.shear(e1=e1, e2=e2)
        elif gal_type == 'sersic':
            sersic_profile = galsim.Sersic(sersicn, half_light_radius = self.gal_sigma)
            self.gal_light = sersic_profile.withFlux(self.gal_flux)
            self.gal_light = self.gal_light.shear(e1=e1, e2=e2)
        elif gal_type == 'bpd':
            bulge = galsim.Sersic(4, half_light_radius=self.bpd_params[0])
            bulge = bulge.shear(e1=bpd_params[1], e2=bpd_params[2])
            disk = galsim.Sersic(1, half_light_radius=self.bpd_params[3])
            disk = disk.shear(e1=bpd_params[4], e2=bpd_params[5])
            bulge_to_total = bpd_params[6]
            self.gal_light = bulge_to_total*bulge + (1-bulge_to_total)*disk

        self.gal_rotate_light = self.gal_light.rotate(90 * galsim.degrees)
        self.gal_light = self.gal_light.shear(g1=g1, g2=g2)
        self.gal_rotate_light = self.gal_rotate_light.shear(g1=g1, g2=g2)

        if not is_self_defined_PSF:
            self.psf_type = psf_type
            self.psf_sigma = psf_sigma
            self.psf_model_sigma = psf_sigma

            if psf_type == 'gaussian':
                self.psf_base = galsim.Gaussian(flux=1.0, sigma=self.psf_sigma)
            elif psf_type == 'kolmogorov':
                self.psf_base  = galsim.Kolmogorov(flux=1.0, half_light_radius=0.5)
                self.psf_base = self.toSize(self.psf_base, self.psf_sigma,weighted=True)
            elif psf_type == 'opticalPSF':
                self.psf_base = galsim.OpticalPSF(1.0, flux=1.0)
                self.psf_base = self.toSize(self.psf_base, self.psf_sigma)
            elif psf_type == 'sersic':
                self.psf_base = galsim.Sersic(psf_sersicn, half_light_radius=1.0)
                self.psf_base = self.toSize(self.psf_base, self.psf_sigma,weighted=True)
            elif psf_type == 'gsobj':
                self.psf_base = psf_gsobj

        else:
            self.psf_type = "self_define"
            truth_image = self_defined_PSF
            truth_psf = galsim.InterpolatedImage(truth_image,scale = pixel_scale)
            truth_measure = galsim.hsm.FindAdaptiveMom(truth_image)
            truth_sigma = truth_measure.moments_sigma
            self.psf_light = truth_psf

    def setup_shapelet_psf(self, m, c, n, bmax = 10):
        self.n = n
        self.sxm = shapeletXmoment(self.psf_base, n, pixel_scale=self.pixel_scale)
        self.psf_light = self.sxm.base_shapelet
        self.psf_model_light = self.sxm.iterative_modify_pq(m, c)
        self.dm = m * self.sxm.get_all_moments(self.sxm.base_psf_image, self.sxm.get_pq_full(n)) + c

    def speed_setup_shapelet_psf(self, m, c, n, psf_light, psf_model_light, dm):
        self.n = n
        self.sxm = shapeletXmoment(self.psf_base, n, pixel_scale=self.pixel_scale)
        self.psf_light = psf_light
        self.psf_model_light = psf_model_light
        self.dm = dm

    def perc_bias(self, metacal=True):
        base_ori_r, base_ori_e = self.measure(metacal=metacal, rot=False, base=True)
        mod_ori_r, mod_ori_e = self.measure(metacal=metacal, rot=False, base=False)
        base_rot_r, base_rot_e = self.measure(metacal=metacal, rot=True, base=True)
        mod_rot_r, mod_rot_e = self.measure(metacal=metacal, rot=True, base=False)
    
        base_g_uncal = np.mean([base_ori_e, base_rot_e], axis=0)
        mod_g_uncal = np.mean([mod_ori_e, mod_rot_e], axis=0)

        R_base = np.mean([base_ori_r, base_rot_r], axis=0)
        R_mod = np.mean([mod_ori_r, mod_rot_r], axis=0)

        g_base_values = self.calibrate_g_values(R_base, base_g_uncal)
        g_mod_values = self.calibrate_g_values(R_mod, mod_g_uncal)
        self.abs_bias = g_mod_values - g_base_values

        return self.abs_bias/self.g

    def calibrate_g_values(self, R, g_uncal_values):
        Rinv = np.linalg.inv(R)
        return np.matmul(Rinv, g_uncal_values)

    def measure(self, metacal=True, rot=False, base=False):
        if base:
            image_epsf = self.psf_light.drawImage(scale=self.pixel_scale)
        else:
            image_epsf = self.psf_model_light.drawImage(scale=self.pixel_scale)

        if rot:
            galaxy = self.gal_rotate_light
        else:
            galaxy = self.gal_light

        final = galsim.Convolve([galaxy,self.psf_light])
        image = final.drawImage(scale = self.pixel_scale)
        if metacal == False:
            shapes = []
            for image in images:
                results = galsim.hsm.EstimateShear(image,image_epsf)
                shape = galsim.Shear(e1=results.corrected_e1, e2=results.corrected_e2)
                shapes.append([shape.g1, shape.g2])

            g_uncal_values = np.array(shapes)
            R_values = np.repeat(np.eye(2)[None, :, :], g_uncal_values.shape[0], axis=0)
            return R_values, g_uncal_values
        else:
            results = self.perform_metacal(image_obs, image_epsf)
            return results["R"], results["g_uncal"]

    def perform_metacal(self, image_obs, image_epsf):
        metacal = mcal.MetacalRunner(image_obs, image_epsf)
        metacal.measure_shear(self.metacal_method)
        results = metacal.get_results()
        return results

    def findAdaptiveSersic(self, sigma, n):
        good_half_light_re = bisect(
            Sersic_sigma,
            sigma/3,
            sigma*5,
            args=(n, self.pixel_scale, sigma)
            )
        return galsim.Sersic(n=n, half_light_radius=good_half_light_re)

    def findAdaptiveKolmogorov(self, sigma):
        good_half_light_re = bisect(
            Kolmogorov_sigma,
            max(self.psf_sigma / 5, self.pixel_scale),
            self.psf_sigma * 5,
            args=(self.pixel_scale,sigma)
            )
        return galsim.Kolmogorov(half_light_radius=good_half_light_re)

    def findAdaptiveOpticalPSF(self, sigma):
        good_fwhm = bisect(
            OpticalPSF_sigma,max(self.psf_sigma / 3, self.pixel_scale),
            self.psf_sigma * 5,
            args=(self.pixel_scale, sigma)
            )
        return galsim.OpticalPSF(good_fwhm)

    def toSize(self, profile, sigma, weighted=True, tol=1e-4):
        if weighted:
            apply_pixel =  max(self.pixel_scale, sigma/10)
            true_sigma = galsim.hsm.FindAdaptiveMom(profile.drawImage(scale =apply_pixel,method = 'no_pixel')).moments_sigma*apply_pixel
        else:
            image = profile.drawImage(scale = self.pixel_scale, method = 'no_pixel')
            true_sigma = image.calculateMomentRadius()

        ratio = sigma/true_sigma
        new_profile = profile.expand(ratio)

        while abs(true_sigma - sigma)>tol:
            ratio = sigma/true_sigma
            new_profile = new_profile.expand(ratio)

            if weighted:
                apply_pixel =  max(self.pixel_scale, sigma/10)
                true_sigma = galsim.hsm.FindAdaptiveMom(new_profile.drawImage(scale =apply_pixel,method = 'no_pixel'),hsmparams=galsim.hsm.HSMParams(max_mom2_iter = 2000)).moments_sigma*apply_pixel
            else:
                #true_sigma = profile.calculateMomentRadius(scale = self.pixel_scale, rtype='trace')
                image = new_profile.drawImage(scale = self.pixel_scale, method = 'no_pixel')
                true_sigma = image.calculateMomentRadius()
        return new_profile

    def real_gal_sigma(self):
        image = self.gal_light.drawImage(scale = self.pixel_scale,method = 'no_pixel')
        return galsim.hsm.FindAdaptiveMom(image).moments_sigma*self.pixel_scale

    def get_actual_dm(self):
        m_truth = self.sxm.get_all_moments(self.psf_light.drawImage(scale=self.pixel_scale), self.sxm.get_pq_full(self.n))
        m_model = self.sxm.get_all_moments(self.psf_model_light.drawImage(scale=self.pixel_scale), self.sxm.get_pq_full(self.n))
        return m_model - m_truth

    def get_gal_trace(self):
        return 2 * self.gal_light.calculateMomentRadius(
            scale=self.pixel_scale,
            rtype='trace'
            )**2

    def get_psf_trace(self):
        return 2 * self.psf_light.calculateMomentRadius(
            scale=self.pixel_scale,
            rtype='trace'
            )**2

    def get_results(self, metacal=True):
        results = dict()

        results['shear_bias'] = self.perc_bias(metacal=metacal)
        results['abs_bias'] = self.abs_bias
        results["gal_type"] = self.gal_type
        results["psf_type"] = self.psf_type
        results["gal_sigma"] = self.gal_sigma # in physical units
        results["psf_sigma"] = self.psf_sigma # in physical units
        results["e1"] = self.e1
        results["e2"] = self.e2
        results["e"] = self.e
        results["sersicn"] = self.sersicn
        results["gal_hlr"] = self.gal_light.calculateHLR()
        results["psf_hlr"] = self.psf_base.calculateHLR()
        results["psf_model_sigma"] = self.psf_model_sigma # in physical units
        results['g'] = self.g
        results["dm"] = self.dm
        results["gal_trace"] = self.get_gal_trace() # in physical units
        results["psf_trace"] = self.get_psf_trace() # in physical units

        # results measured on images
        actual_dm, actual_p, gal_p = self.get_measured_quantities()
        results["actual_dm"] = actual_dm
        results["actual_dp"] = {k:p for k,p in actual_p.items() if 'd' in k}
        results["psf_p"] = {k:p for k,p in actual_p.items() if 'd' not in k}
        results["gal_p"] = gal_p

        return results
