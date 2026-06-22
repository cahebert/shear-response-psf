import numpy as np
import galsim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import momhelper

try:
    import ngmix
    HAS_NGMIX = True
except ImportError:
    HAS_NGMIX = False
    print("WARNING: ngmix not importable.")

try:
    import piff
    HAS_PIFF = True
except ImportError:
    HAS_PIFF = False
    print("WARNING: piff not importable.")

def hsm_moments(image, pixel_scale):
    """Second moments + T4/rho4 via galsim.hsm.FindAdaptiveMom.

    T4   = rho4 * sigma^2 = rho4 * T/2  (arcsec^2, comparable to ngmix/piff T4)
    rho4 = moments_rho4 (dimensionless kurtosis; = 2 for a Gaussian)
    """
    res      = galsim.hsm.FindAdaptiveMom(image)
    sigma_px = res.moments_sigma
    T        = 2.0 * (sigma_px * pixel_scale) ** 2
    shape    = res.observed_shape
    rho4     = res.moments_rho4
    T4       = rho4 * T / 2.0   # rho4 * sigma^2
    return dict(T=T, e1=shape.e1, e2=shape.e2,
                T4=T4, rho4=rho4, method='galsim.hsm')

def home_moments(image, pixel_scale):
    """Second moments + T4/rho4 via galsim.hsm.FindAdaptiveMom.
    
    T4   = rho4 * sigma^2 = rho4 * T/2  (arcsec^2, comparable to ngmix/piff T4)
    rho4 = moments_rho4 (dimensionless kurtosis; = 2 for a Gaussian)
    """
    import shapelets
    sxm = shapelets.shapeletXmoment(galsim.Gaussian(fwhm=3), 4, pixel_scale=pixel_scale)
    pqlist = sxm.get_pq_full(sxm.n)
    moms = sxm.get_all_moments(image, pqlist)
    params = shapelets.get_params_from_moments(moms, pixel_scale=pixel_scale)
    params.update(method='home')
    return params

def ngmix_moments(image, pixel_scale):
    """
    Second + fourth moments via ngmix.gaussmom with a weight matched to the HSM size.
    Fourth-order quantity: res['m22'] / res['m11'] (spin-0 fourth / second moment).
    """
    if not HAS_NGMIX:
        return None
    ny, nx = image.array.shape
    cen    = ((ny - 1) / 2.0, (nx - 1) / 2.0)
    jac    = ngmix.DiagonalJacobian(scale=pixel_scale, row=cen[0], col=cen[1])
    obs    = ngmix.Observation(image.array, jacobian=jac)

    # Weight FWHM matched to HSM adaptive size
    hsm_T  = hsm_moments(image, pixel_scale)['T']
    fwhm_w = ngmix.moments.T_to_fwhm(hsm_T)  # arcsec
    # fwhm_w = 2.355 * np.sqrt(hsm_T / 2.0)   # arcsec

    try:
        from ngmix import gaussmom as gm
        gauss = gm.GaussMom(fwhm_w, with_higher_order=True)
        res = gauss.go(obs)
        if res['flags'] == 0:
            res['T'] *= 2 # de-weight
            T4   = res['M22'] / res['M11']   # arcsec^2; m11 = sigma^2 = T/2
            rho4 = T4 / res['M11']      # = T4 / sigma^2, dimensionless
            return dict(T=res['T'], e1=res['e1'], e2=res['e2'],
                        T4=T4, rho4=rho4, method='ngmix.GaussMom')
    except Exception as err:
        print(f"  ngmix GaussMom failed: {err}")
    return None


def piff_moments(image, pixel_scale):
    """
    Second + fourth moments via piff.stats.measureShapes(fourthOrder=True).
    shapes[0] = [e1, e2, T, T4]; T4 (index 3) is the spin-0 fourth-order moment.

    NOTE: if this raises, check piff.StarData / piff.Star constructor signature
    for the installed version.
    """
    if not HAS_PIFF:
        return None
    try:
        ny, nx    = image.array.shape
        image_pos = galsim.PositionD(nx / 2.0, ny / 2.0)
        star_data = piff.StarData(image, image_pos, properties={})
        star      = piff.Star(star_data, None)
        stats     = piff.stats.Stats()
        _, shapes, _ = stats.measureShapes(stars=[star], psf=None, fourth_order=True)
        s    = shapes[0]
        T    = s[3]
        T4   = s[7]
        rho4 = T4 * 2.0 / T   # = T4 / sigma^2, dimensionless
        return dict(T=T, e1=s[4], e2=s[5],
                    T4=T4, rho4=rho4, method='piff.stats.measureShapes')
    except Exception as err:
        print(f"  piff moment measurement failed: {err}")
        # print("  Check piff.StarData / piff.Star constructor for your version.")
    return None


def test_moment_measurements():
    """"""
    stamp_size = 55
    pixel_scale = 0.2
    beta = 3.5
    fwhm = 0.9

    moffat = galsim.Moffat(beta=beta, fwhm=fwhm)
    gauss = galsim.Gaussian(fwhm=fwhm)

    print("-" * 100)
    print(f"{'Profile':<28}  {'Method':<30}  {'T':>8}  {'e1':>8}  {'e2':>8}  {'T4 (arcsec²)':>14}  {'rho4':>8}")
    print("-" * 100)

    for profile, profile_str in zip([moffat, gauss], ['moffat','gaussian']):
        image = profile.drawImage(
            nx=stamp_size, ny=stamp_size,
            scale=pixel_scale, method='no_pixel'
        )
        # profile_str = f'{Moffat} β={beta}, {t_label}'

        results = [
            hsm_moments(image, pixel_scale=pixel_scale),
            ngmix_moments(image, pixel_scale=pixel_scale),
            piff_moments(image, pixel_scale=pixel_scale),
            home_moments(image, pixel_scale=pixel_scale),
        ]

        first = True
        for res in results:
            if res is None:
                continue
            prefix = profile_str if first else ''
            first  = False
            try:
                print(f"{prefix:<28}  {res['method']:<30}  "
                    f"{res['T']:8.4f}  {res['e1']:8.5f}  {res['e2']:8.5f}  "
                    f"{res['T4']:14.5f}  {res['rho4']:8.4f}")
            except KeyError:
                print(f"{prefix:<28}  {res['method']:<30}  "
                    f"{res['t2']:8.4f}  {res['e2_1']:8.5f}  {res['e2_2']:8.5f}  "
                    f"{res['t4']:14.5f}  {res['rho4']:8.4f}")
        print()

if __name__ == '__main__':
    test_moment_measurements()