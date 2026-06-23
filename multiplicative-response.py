import numpy as np
import galsim
import ngmix

import shapelets

def do_tests_speed(tests, test_m, test_c, n, debug=False):
    testsresult=[]
    for i in range(len(tests)):
        test = shapelets.HOMExShapeletPair(*tests[i][:-1],**tests[i][-1])
        if i!=0:
            test.speed_setup_shapelet_psf(test_m[i], test_c[i], n, psf_light, psf_model_light, dm)
        else:
            test.setup_shapelet_psf(test_m[i], test_c[i], n)
            psf_light = test.psf_light
            psf_model_light = test.psf_model_light
            dm = test.dm

        results = test.get_results(metacal=True)
        testsresult.append(results)

        if debug: print(results['actual_dp'])

    if debug:
        scale = tests[0][-1]['pixel_scale']
        test.n_pix
        model_img = psf_model_light.drawImage(
            nx=test.n_pix, ny=test.n_pix, scale=scale
        )
        psf_img = psf_light.drawImage(
            nx=test.n_pix, ny=test.n_pix, scale=scale
        )

    return testsresult, (psf_img.array, model_img.array) if debug else None

def e2(e1,e):
    return np.sqrt(e**2 - e1**2)

n_shapelet = 4
if n_shapelet == 4:
    n_moments = 12
elif n_shapelet == 6:
    n_moments = 25

def add_T2(c_list, delta, psf_sigma, pixel_scale):
    psf_sigma_pix = psf_sigma / pixel_scale
    delta_pix = delta / pixel_scale**2
    
    delta_sigma = delta_pix / (4 * psf_sigma_pix)
    c_list[1] += delta_sigma

def add_T4(m_list, delta, psf_sigma, pixel_scale, rho4=2):
    """Add m40, m04, m22 amounts of error corresponding to delta T4 error."""
    # add error in pixels (should cancel out anyway)
    m11_w_pix = (psf_sigma / pixel_scale)**2 
    delta_pix = delta / pixel_scale**2
    
    delta_rho4 = delta_pix / m11_w_pix
    dr4r4 = delta_rho4 / rho4
    # inject drho/rho into m
    m_list[7] += dr4r4
    m_list[9] += dr4r4
    m_list[11] += dr4r4

def run_mult_tests(
    t_order, sigma_gal=[1.5], gal_shear=0.01, delta_t=0.001, 
    debug=False, psf_sigma=1.5, mcal_method='estimateShear',
    pixel_scale=1.0,
    ):
    """Run response tests with PSF size perturbations and zero PSF shape."""
    if gal_shear==0: gal_shear = 1e-8

    N = 40
    m_T = np.zeros(shape=(N,n_moments))
    c_T = np.zeros(shape=(N,n_moments))
    if t_order == 2:
        for size_index in range(N):
            add_T2(c_T[size_index], delta_t, psf_sigma, pixel_scale)
    elif t_order == 4:
        for size_index in range(N):
            add_T4(m_T[size_index], delta_t, psf_sigma, pixel_scale, rho4=2)

    config_base = [(
        "gaussian", sgal,
        0.0, 0.0, gal_shear, 1e-8,
        "gaussian", psf_sigma,
        {'subtract_intersection': True,
         'metacal_method': mcal_method,
         'pixel_scale': pixel_scale}
        ) for sgal in sigma_gal]

    results_base, psf_model = do_tests_speed(
        config_base, m_T, c_T, n_shapelet, debug=debug
        )

    if debug:
        import matplotlib.pyplot as plt

        f,a = plt.subplots(1,3, figsize=(9,3), sharex=True, sharey=True)
        m=a[0].imshow(
            psf_model[0],
            vmin=-np.max(psf_model[0]), vmax=np.max(psf_model[0]),
            cmap='RdBu', origin='lower'
            )
        a[0].set_title(r'PSF')
        plt.colorbar(m, ax=a[0])
        a[0].grid(alpha=0.25, color='grey')
        m=a[1].imshow(
            psf_model[1],
            vmin=-np.max(psf_model[0]), vmax=np.max(psf_model[0]),
            cmap='RdBu', origin='lower'
            )
        a[1].set_title(r'PSF$_{+\Delta}$')
        plt.colorbar(m, ax=a[1])
        a[1].grid(alpha=0.25, color='grey')
        m=a[2].imshow(
            psf_model[0] - psf_model[1],
            vmin=-np.max(psf_model[0] - psf_model[1]), vmax=np.max(psf_model[0] - psf_model[1]),
            cmap='RdBu', origin='lower'
            )
        a[2].set_title(r'$\Delta$PSF')
        plt.colorbar(m, ax=a[2])
        a[2].grid(alpha=0.25, color='grey')

        plt.tight_layout()
        plt.show()

    # get size ratio from the results
    size_ratio = np.array([t["psf_p"]['t2']/t['gal_p']['t2'] for t in results_base])

    if t_order == 2:
        dT_over_T = np.array([t["actual_dp"]['dt2/t2'] for t in results_base])
    if t_order == 4:
        dT_over_T = np.array([t["actual_dp"]['dt4/t4'] for t in results_base])
    
    dg = np.array([t["abs_bias"][0] for t in results_base])
    # sigma_dg = np.array([t["abs_bias_std"][0] for t in results_base])

    return size_ratio, dg, dT_over_T


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug', action='store_true',
                           help='Turn on debug mode to get difference images')
    args = argparser.parse_args()

    import scipy
    import ngmix
    
    pscale = 0.2
    sigma_psf_pix = 1.75 * 2
    # size_ratio = np.sqrt(np.linspace(0.05, 4, 10))
    size_ratio = np.sqrt([0.15, 0.5, 1, 1.5, 2])
    
    sigma_psf = sigma_psf_pix * pscale
    sigma_gal = sigma_psf / size_ratio

    t_psf = ngmix.moments.fwhm_to_T(ngmix.moments.sigma_to_fwhm(sigma_psf))
    t_gal = ngmix.moments.fwhm_to_T(ngmix.moments.sigma_to_fwhm(sigma_gal))
    t_ratio_plt = t_psf / t_gal
    
    goal_dtt = np.array([3.6e-4, 3.3e-4, -6.7e-3, -6e-3])
    # goal_dtt = np.array([5e-4])
    goal_dt = goal_dtt * t_psf

    # shears = np.array([-0.02, -0.01, 0, 0.01, 0.02])
    shear = 0.02

    results = {}
    if args.debug:
        images = np.zeros((2,200,200))
        order = {'2':0, '4':1}

    for i,dtt in enumerate(goal_dtt):
        print(dtt)
        results[i] = {}
        for t_order in [2,4]:
            if t_order == 2: dt = dtt * t_psf
            if t_order == 4: dt = dtt * 2 * (sigma_psf**2)  # bc t4 = m11 * rho4, m11 = sigma^2, and rho4 = 2 for gaussian
            t_ratio, dg, dT_over_T = run_mult_tests(
                t_order,
                sigma_gal=sigma_gal,
                gal_shear=shear,
                delta_t=dt,
                debug=args.debug,
                psf_sigma=sigma_psf,
                mcal_method='estimateShear',
                pixel_scale=pscale
                )
            results[i]['tratio_' + str(t_order)] = t_ratio
            results[i]['dg_' + str(t_order)] = dg
            # results[i]['sigdg_' + str(t_order)] = sigma_dg
            results[i]['dtt_' + str(t_order)] = dT_over_T

    import matplotlib.pyplot as plt
    f,a = plt.subplots(2, 1, sharex=True, sharey=True)

    for (k, res), color in zip(results.items(), ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']):
        # tratio_plt = t_ratio_plt
        tratio_plt = res['tratio_2']
        a[0].errorbar(
            tratio_plt,
            res['dg_2'] / shear,
            # yerr=res['sigdg_2'] / shear,
            fmt='o', color=color, label=rf'{res['dtt_2'][0]:.2g}'
            )
        a[0].plot(tratio_plt, res['dtt_2'] * tratio_plt, '-', color=color)
            # label=r'Gaussian expectation $m_{(2)}=-\frac{Tpsf}{Tgal} * \frac{dT}{T}$'

    # for dt, color in zip(goal_dt[2:], ['tab:blue', 'tab:orange']):
        a[1].errorbar(
            res['tratio_4'],
            res['dg_4'] / shear,
            # yerr=res['sigdg_4'] / shear,
            fmt='o', color=color,
            label=rf'{res['dtt_4'][0]:.2g}'
        )
        # a[1].plot(tratio_plt, -res['dtt_4'] * tratio_plt, '-', color=color)

    a[0].legend(title=r'$\Delta T^{(2)}/T^{(2)}$')
    a[1].legend(title=r'$\Delta T^{(4)}/T^{(4)}$')
    a[0].set_ylabel(r'$(g_{\rm \Delta PSF} - g) / g$')
    a[1].set_ylabel(r'$(g_{\rm \Delta PSF} - g) / g$')
    a[1].set_xlabel(r'$T_{\rm PSF}/T_{\rm gal}$')
    plt.tight_layout()
    plt.savefig('m_vs_Tratio_paperdtt.png', dpi=150)
    plt.show()

    # f,a = plt.subplots(2,1, sharex=True)
    # a[0].plot(shears, results['dg_2'] / shears,'o')
    # a[0].plot(shears, -results['dtt_2'] * t_ratio, '-')
    # # a[0].set_ylabel(r'$\delta g/ \delta T^{(2)}$')
    # a[1].plot(shears, results['dg_4'] / shears,'o')
    # # a[1].plot(shears, -results['dtt_4'] * t_ratio,'-')
    # # a[1].set_ylabel(r'$\delta g/ \delta T^{(4)}$')
    # a[1].set_xlabel('input shear')
    # plt.tight_layout()
    # plt.show()
