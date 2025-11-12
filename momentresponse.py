import numpy as np
import galsim
import ngmix

import shapelets

def do_tests_speed(tests,test_m, test_c, n, debug=False):
    testsresult=[]
    for i in range(len(tests)):
        test = shapelets.HOMExShapeletPair(*tests[i][:-1],**tests[i][-1])
        if i!=0:
            test.speed_setup_shapelet_psf(test_m[i],test_c[i],n,psf_light, psf_model_light, dm)

            if debug:
                model_img = psf_model_light.drawImage(nx=200, ny=200, scale=0.1)
                psf_img = psf_light.drawImage(nx=200, ny=200, scale=0.1)
                diff = psf_img.array - model_img.array

                # model_e4_1 = test.sxm.moment_measure(model_img, 4, 0)
                # model_e4_1 -= test.sxm.moment_measure(model_img, 0, 4)
                # psf_e4_1 = test.sxm.moment_measure(psf_img, 4, 0)
                # psf_e4_1 -= test.sxm.moment_measure(psf_img, 0, 4)
                # print(model_e4_1, psf_e4_1, model_e4_1-psf_e4_1)

                # model_e4_2 = 2*test.sxm.moment_measure(model_img, 1, 3)
                # model_e4_2 += 2*test.sxm.moment_measure(model_img, 3, 1)
                # psf_e4_2 = 2*test.sxm.moment_measure(psf_img, 1, 3)
                # psf_e4_2 += 2*test.sxm.moment_measure(psf_img, 3, 1)
                # print(model_e4_2, psf_e4_2, model_e4_2-psf_e4_2)

                # model_T4 = test.sxm.moment_measure(model_img, 4, 0) + test.sxm.moment_measure(model_img, 0, 4)
                # model_T4 += 2*test.sxm.moment_measure(model_img, 2, 2)
                # psf_T4 = test.sxm.moment_measure(psf_img, 4, 0) + test.sxm.moment_measure(psf_img, 0, 4)
                # psf_T4 += 2*test.sxm.moment_measure(psf_img, 2, 2)
                # print(model_T4, psf_T4, model_T4 - psf_T4)

                # if abs(model_e4_1 - psf_e4_1) > 1e-4:
                #     print(f"Warning: e4_1 difference is {model_e4_1 - psf_e4_1} at test {i}")
                # if abs(model_e4_2 - psf_e4_2) > 1e-4:
                #     print(f"Warning: e4_2 difference is {model_e4_2 - psf_e4_2} at test {i}")
                # psf_shape_mag = 0.005
                # if abs(model_e4_1 - psf_shape_mag) > 1e-4:
                #     print(f"Warning: model e4_1 not equal to input! {model_e4_1 - psf_shape_mag} at test {i}")
        else:
            test.setup_shapelet_psf(test_m[i],test_c[i],n)
            psf_light = test.psf_light
            psf_model_light = test.psf_model_light
            dm = test.dm
        results = test.get_results(metacal = True)
        testsresult.append(results)
    return testsresult, diff if debug else None

def e2(e1,e):
    return np.sqrt(e**2 - e1**2)

n_shapelet = 4
if n_shapelet == 4:
    n_moments = 12
elif n_shapelet == 6:
    n_moments = 25

def add_e4_1(c_list, delta):
    c_list[7] -= delta / 2
    c_list[11] += delta / 2

def add_e4_2(c_list, delta):
    c_list[8] += delta / 4
    c_list[10] += delta / 4

def add_e2_1(c_list, delta):
    c_list[2] += delta

def add_e2_2(c_list, delta):
    c_list[0] += delta

def add_T2(c_list, delta, psf_sigma):
    # dsigma = dT / (4 * sigma)
    delta_sigma = delta / (4 * psf_sigma)
    c_list[1] += delta_sigma

def add_T4(m_list, delta, psf_sigma, rho4=2):
    # t4 = m11 * rho4, and m11 = T/2 = sigma**2
    delta_rho4 = delta / (psf_sigma**2)
    # inject drho/rho into m
    m_list[7] += delta_rho4 / rho4
    m_list[9] += delta_rho4 / rho4
    m_list[11] += delta_rho4 / rho4

def run_eta_tests(e_order, t_order, delta_t=0.001, debug=False):
    """Run response tests with PSF size perturbations and nonzero PSF shape."""
    psf_shape_mag = 0.005
    psf_sigma = 1.5
    if e_order == 2:
        ## set up PSFs with non-zero g2 components ##
        psf_real = galsim.Gaussian(sigma=psf_sigma).shear(e1=psf_shape_mag, e2=0.0)
        psf_complex = galsim.Gaussian(sigma=psf_sigma).shear(e1=0.0, e2=psf_shape_mag)

    elif e_order == 4:
        config_psf = ["gaussian", 1.5, 0.0, 0.0, 1e-8, 1e-8, "gaussian", 1.5, {'subtract_intersection':True}]
        ## set up PSFs with non-zero e4 components ##
        test_real = shapelets.HOMExShapeletPair(*config_psf[:-1], **config_psf[-1])
        test_complex = shapelets.HOMExShapeletPair(*config_psf[:-1], **config_psf[-1])

        m_psf = np.zeros(shape=(n_moments))
        c_psf = np.zeros(shape=(n_moments))
        add_e4_1(c_psf, psf_shape_mag)

        test_real.setup_shapelet_psf(m_psf, c_psf, n_shapelet)
        psf_real = test_real.psf_model_light
        psf_r_img = psf_real.drawImage(scale=0.1, nx=200, ny=200)
        e4_1 = -test_real.sxm.moment_measure(psf_r_img, q=4, p=0)
        e4_1 += test_real.sxm.moment_measure(psf_r_img, q=0, p=4)

        m_psf = np.zeros(shape=(n_moments))
        c_psf = np.zeros(shape=(n_moments))
        add_e4_2(c_psf, psf_shape_mag)

        test_complex.setup_shapelet_psf(m_psf, c_psf, n_shapelet)
        psf_complex = test_complex.psf_model_light
        psf_c_img = psf_complex.drawImage(scale=0.1, nx=200, ny=200)
        e4_2 = 2*test_complex.sxm.moment_measure(psf_c_img, q=1, p=3)
        e4_2 += 2*test_complex.sxm.moment_measure(psf_c_img, q=3, p=1)

        if abs(e4_2 - psf_shape_mag) > 5e-3:
            print('e4 mag not equal to input, something is wrong!')
            print(f"in: {psf_shape_mag}, out: {e4_2}")
        if abs(e4_1 - psf_shape_mag) > 5e-3:
            print('e4 mag not equal to input, something is wrong!')
            print(f"in: {psf_shape_mag}, out: {e4_1}")
        if abs(e4_1 - e4_2)/e4_2 > 1e-3:
            print('e4_1 and e4_2 are not equal, something is wrong!')
            print(f"e4_1: {e4_1}, e4_2: {e4_2}")
        # use the actual measured shape for the dc calculation to get norm correct.
        psf_shape_mag = np.mean([e4_1, e4_2])
    # for size residual
    N = 40
    m_T = np.zeros(shape=(N,n_moments))
    c_T = np.zeros(shape=(N,n_moments))
    if t_order == 2:
        for size_index in range(N):
            add_T2(c_T[size_index], delta_t, psf_sigma)
    elif t_order == 4:
        for size_index in range(N):
            add_T4(m_T[size_index], delta_t, psf_sigma, rho4=2)

    config_eta_real = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gsobj", psf_sigma,
        {'subtract_intersection':True, 'psf_gsobj':psf_real}
        ) for sgal in sigma_gal]

    config_eta_complex = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gsobj", psf_sigma,
        {'subtract_intersection':True, 'psf_gsobj':psf_complex}
        ) for sgal in sigma_gal]

    results_eta_real, diff_real = do_tests_speed(
        config_eta_real, m_T, c_T, n_shapelet, debug=debug
        )
    results_eta_complex, diff_complex = do_tests_speed(
        config_eta_complex, m_T, c_T, n_shapelet, debug=debug
        )

    if debug:
        import matplotlib.pyplot as plt
        from scipy.ndimage import rotate

        f,a = plt.subplots(1,3, figsize=(8,2.5), sharex=True, sharey=True)
        m=a[0].imshow(
            diff_real, vmin=-np.max(diff_real), vmax=np.max(diff_real),
            cmap='RdBu', origin='lower'
            )
        a[0].set_title(rf'Re($e${e_order}), $T${t_order}')
        plt.colorbar(m, ax=a[0])
        a[0].grid(alpha=0.25, color='grey')

        m=a[1].imshow(
            diff_complex, vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
            cmap='RdBu', origin='lower'
            )
        a[1].set_title(rf'Im($e${e_order}), $T${t_order}')
        plt.colorbar(m, ax=a[1])
        a[1].grid(alpha=0.25, color='grey')

        m=a[2].imshow(
            diff_real-diff_complex,
            vmin=-np.max(diff_real-diff_complex),
            vmax=np.max(diff_real-diff_complex),
            cmap='RdBu', origin='lower'
            )
        a[2].set_title(r'$\Delta$')
        plt.colorbar(m, ax=a[2])
        a[2].grid(alpha=0.25, color='grey')
        plt.tight_layout()
        plt.show()

        f,a = plt.subplots(2,2, figsize=(6,6), sharex=True, sharey=True)
        m=a[0,0].imshow(
            diff_real,
            #vmin=-np.max(diff_real), vmax=np.max(diff_real),
            cmap='RdBu', origin='lower'
            )
        a[0,0].set_title(r'$\Delta$PSF$_{e1}$')
        plt.colorbar(m, ax=a[0,0])
        a[0,0].grid(alpha=0.25, color='grey')
        m=a[0,1].imshow(
            rotate(diff_real, angle=45, reshape=False),
            #vmin=-np.max(diff_real), vmax=np.max(diff_real),
            cmap='RdBu', origin='lower'
            )
        a[0,1].set_title(r'rot($\Delta$PSF$_{e1}$)')
        plt.colorbar(m, ax=a[0,1])
        a[0,1].grid(alpha=0.25, color='grey')

        m=a[1,0].imshow(
            diff_complex, #vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
            cmap='RdBu', origin='lower'
            )
        a[1,0].set_title(r'$\Delta$PSF$_{e2}$')
        plt.colorbar(m, ax=a[1,0])
        a[1,0].grid(alpha=0.25, color='grey')
        m=a[1,1].imshow(
            rotate(diff_complex, angle=45, reshape=False),
            #vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
            cmap='RdBu', origin='lower'
            )
        a[1,1].set_title(r'rot($\Delta$PSF$_{e2}$)')
        plt.colorbar(m, ax=a[1,1])
        a[1,1].grid(alpha=0.25, color='grey')
        plt.tight_layout()
        plt.show()

    # get size ratio and eta from the results
    size_ratio = np.array([t['psf_sigma']/t['gal_sigma'] for t in results_eta_real])**2

    if t_order == 2:
        t_psf_mag = ngmix.moments.fwhm_to_T(ngmix.moments.sigma_to_fwhm(1.5))
    elif t_order == 4:
        t_psf_mag = 2 * psf_sigma**2  # kurtosis = 2 for gaussian, and T4 = m11 * rho4
    dT_over_T = delta_t / t_psf_mag

    # minus sign bc my residual is obs - model
    eta_real = -np.array([t["abs_bias"][0] for t in results_eta_real]) / (dT_over_T * psf_shape_mag)
    eta_complex = -np.array([t["abs_bias"][1] for t in results_eta_complex]) / (dT_over_T * psf_shape_mag)

    return size_ratio, eta_real, eta_complex, diff_real

def run_beta_tests(e_order, delta_e=0.001, debug=False):
    """Run response tests with PSF shape perturbations only."""
    psf_sigma = 1.5

    N = 40
    m_e = np.zeros(shape=(N,n_moments))
    c_e1 = np.zeros(shape=(N,n_moments))
    c_e2 = np.zeros(shape=(N,n_moments))
    if e_order == 2:
        for size_index in range(N):
            add_e2_1(c_e1[size_index], delta_e)
            add_e2_2(c_e2[size_index], delta_e)
    elif e_order == 4:
        for size_index in range(N):
            add_e4_1(c_e1[size_index], delta_e)
            add_e4_2(c_e2[size_index], delta_e)

    config_beta_real = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gaussian", psf_sigma,
        {'subtract_intersection':True}
        ) for sgal in sigma_gal]

    config_beta_complex = [(
        "gaussian", sgal,
        0.0, 0.0, 1e-8, 1e-8,
        "gaussian", psf_sigma,
        {'subtract_intersection':True}
        ) for sgal in sigma_gal]

    results_beta_real, diff_real = do_tests_speed(
        config_beta_real, m_e, c_e1, n_shapelet, debug=debug
        )
    results_beta_complex, diff_complex = do_tests_speed(
        config_beta_complex, m_e, c_e2, n_shapelet, debug=debug
        )

    if debug:
        import matplotlib.pyplot as plt
        f,a = plt.subplots(1,3, figsize=(8,2.5), sharex=True, sharey=True)

        m=a[0].imshow(diff_real, vmin=-np.max(diff_real), vmax=np.max(diff_real),
                      cmap='RdBu', origin='lower')
        a[0].set_title(rf'Re($e${e_order})')
        plt.colorbar(m, ax=a[0])
        a[0].grid(alpha=0.25, color='grey')

        m=a[1].imshow(diff_complex, vmin=-np.max(diff_complex), vmax=np.max(diff_complex),
                      cmap='RdBu', origin='lower')
        a[1].set_title(rf'Im($e${e_order})')
        plt.colorbar(m, ax=a[1])
        a[1].grid(alpha=0.25, color='grey')

        m=a[2].imshow(diff_real-diff_complex,
                      vmin=-np.max(diff_real-diff_complex),
                      vmax=np.max(diff_real-diff_complex),
                      cmap='RdBu', origin='lower')
        a[2].set_title(r'$\Delta$')
        plt.colorbar(m, ax=a[2])
        a[2].grid(alpha=0.25, color='grey')
        plt.tight_layout()
        # plt.savefig('/Users/clairealice/Desktop/e{}_realvcomplex.png'.format(e_order), dpi=200)
        plt.show()

    # get size ratio and eta from the results
    size_ratio = np.array([t['psf_sigma']/t['gal_sigma'] for t in results_beta_real])**2

    # minus sign to keep sign convenction of residual the same as DES vs HSC
    beta_real = -np.array([t["abs_bias"][0] for t in results_beta_real]) / delta_e
    beta_complex = -np.array([t["abs_bias"][1] for t in results_beta_complex]) / delta_e

    return size_ratio, beta_real, beta_complex, diff_real


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug', action='store_true',
                           help='Turn on debug mode to get difference images')
    args = argparser.parse_args()

    import scipy

    size_ratio = np.sqrt(np.linspace(0.01, 3, 30))
    sigma_gal = 1.5 / size_ratio

    results = {}
    if args.debug:
        images = np.zeros((6,200,200))
        order = {'2':0, '4':1, '22':2, '44':3,'24':4, '42':5 }

    for e_order in [2,4]:
        print(f"Running beta{e_order} tests...")
        size_ratio, beta_r, beta_c, diff_img_r = run_beta_tests(e_order, delta_e=0.001)
        results['beta' + str(e_order) + '_r'] = list(beta_r)
        results['beta' + str(e_order) + '_c'] = list(beta_c)
        if args.debug:
            images[order[str(e_order)]] = diff_img_r

        for t_order in [2,4]:
            print(f"Running eta{e_order}{t_order} tests...")
            _, eta_r, eta_c, diff_img_r = run_eta_tests(e_order, t_order, delta_t=0.0005)
            results['eta' + str(e_order) + str(t_order) + '_r'] = list(eta_r)
            results['eta' + str(e_order) + str(t_order) + '_c'] = list(eta_c)
            if args.debug:
                images[order[str(e_order)+str(t_order)]] = diff_img_r
    results['size_ratio'] = list(size_ratio)

    if args.debug:
        np.save('test_images.npy', images)
    import json
    with open('moment_response_tratio_results_mcal.json', 'w') as f:
        json.dump(results, f)
