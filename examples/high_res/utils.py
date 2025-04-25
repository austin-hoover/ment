import numpy as np


def gen_ellipses(name: str) -> list[list[float]]:
    ellipses = None
    name = name.lower()
    if name == "shepp-logan":
    	ellipses = gen_ellipses_shepp_logan()
    elif name == "mod-shepp-logan":
        ellipses = gen_ellipses_mod_shepp_logan()
    else:
        raise ValueError ("Unknown phantom type: %s" % name)

    return ellipses


def gen_ellipses_shepp_logan():
	#  Standard head phantom, taken from Shepp & Logan
	return [[   2,   .69,   .92,    0,      0,   0],
	        [-.98, .6624, .8740,    0, -.0184,   0],
	        [-.02, .1100, .3100,  .22,      0, -18],
	        [-.02, .1600, .4100, -.22,      0,  18],
	        [ .01, .2100, .2500,    0,    .35,   0],
	        [ .01, .0460, .0460,    0,     .1,   0],
	        [ .02, .0460, .0460,    0,    -.1,   0],
	        [ .01, .0460, .0230, -.08,  -.605,   0],
	        [ .01, .0230, .0230,    0,  -.606,   0],
	        [ .01, .0230, .0460,  .06,  -.605,   0]]

def gen_ellipses_mod_shepp_logan():
	#  Modified version of Shepp & Logan's head phantom, 
	#  adjusted to improve contrast.  Taken from Toft.
	return [[   1,   .69,   .92,    0,      0,   0],
	        [-.80, .6624, .8740,    0, -.0184,   0],
	        [-.20, .1100, .3100,  .22,      0, -18],
	        [-.20, .1600, .4100, -.22,      0,  18],
	        [ .10, .2100, .2500,    0,    .35,   0],
	        [ .10, .0460, .0460,    0,     .1,   0],
	        [ .10, .0460, .0460,    0,    -.1,   0],
	        [ .10, .0460, .0230, -.08,  -.605,   0],
	        [ .10, .0230, .0230,    0,  -.606,   0],
	        [ .10, .0230, .0460,  .06,  -.605,   0]]

    
def gen_phantom(res: int = 256, name: str = "mod-shepp-logan", ellipses=None) -> np.ndarray:
    """Create a Shepp-Logan or modified Shepp-Logan phantom.

    A phantom is a known object (either real or purely mathematical)
    that is used for testing image reconstruction algorithms.  The
    Shepp-Logan phantom is a popular mathematical model of a cranial
    slice, made up of a set of ellipses.  This allows rigorous
    testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform (see the
    function `radon').

    Inputs
    ------
    res : The edge length of the square image to be produced.

    name : The type of phantom to produce. Either
      "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
      if `ellipses' is also specified.

    ellipses : Custom set of ellipses to use.  These should be in
      the form
            [[I, a, b, x0, y0, phi],
             [I, a, b, x0, y0, phi],
             ...]
      where each row defines an ellipse.
      I : Additive intensity of the ellipse.
      a : Length of the major axis.
      b : Length of the minor axis.
      x0 : Horizontal offset of the centre of the ellipse.
      y0 : Vertical offset of the centre of the ellipse.
      phi : Counterclockwise rotation of the ellipse in degrees,
            measured as the angle between the horizontal axis and
            the ellipse major axis.
      The image bounding box in the algorithm is [-1, -1], [1, 1],
      so the values of a, b, x0, y0 should all be specified with
      respect to this box.

    Output
    ------
    P : A phantom image.

    Usage example
    -------------
      import matplotlib.pyplot as pl
      P = phantom ()
      pl.imshow (P)

    References
    ----------
    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.

    """
    if ellipses is None:
        ellipses = gen_ellipses(name)
    elif np.size(ellipses, 1) != 6:
        raise AssertionError("Wrong number of columns in user phantom")

    # Blank image
    n = res
    p = np.zeros((n, n))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1 : 1 : (1j * n), -1 : 1 : (1j * n)]

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1] ** 2
        b2 = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        locs = (((x * cos_p + y * sin_p) ** 2) / a2 + ((y * cos_p - x * sin_p) ** 2) / b2) <= 1

        # Add the ellipse intensity to those pixels
        p[locs] += I

    return p


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = gen_phantom()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pcolormesh(image.T, cmap="Greys")
    plt.show()