

def get_mij_mapping():
    # index order in dm is the same as the order of pq in self.sxm.get_pq_full(n)
    # which is [(0,2), (1,1), (2,0), (0,3), (1,2), (2,1), (3,0), (0,4), (1,3), (2,2), (3,1), (4,0)]
    mapping = {
        'm02': 0,
        'sig': 1,
        'm20': 2,
        'm03': 3,
        'm12': 4,
        'm21': 5,
        'm30': 6,
        'm04': 7,
        'm13': 8,
        'm22': 9,
        'm31': 10,
        'm40': 11
    }
    return mapping

def get_params_from_moments(m, pixel_scale=None):
    mmap = get_mij_mapping()
    if pixel_scale is None:
        ps = 1
    else:
        ps = pixel_scale
    t2 = 2 * (m[mmap['sig']] * ps)**2
    rho4 = m[mmap['m40']] + m[mmap['m04']] + 2*m[mmap['m22']]
    
    t4 = rho4 * t2 / 2  # m11_w = t2/2

    e2_1 = (m[mmap['m20']] * ps) / t2
    e2_2 = (m[mmap['m02']] * ps) / t2
    e4_1 = (m[mmap['m40']] - m[mmap['m04']]) * ps**2 / (t2 / 2)**2
    e4_2 = 2 * (m[mmap['m31']] + m[mmap['m13']]) * ps**2 / (t2 / 2)**2
    
    return {
        't2': t2,
        'rho4': rho4,
        't4': t4,
        'e2_1': e2_1,
        'e2_2': e2_2,
        'e4_1': e4_1,
        'e4_2': e4_2,
    }
