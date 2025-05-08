import numpy as np
import scipy
import scipy.linalg
from scipy.special import eval_chebyt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import itertools
import warnings


class ChIMES:
    """
    Python Implementation of a subset of ChIMES module in [ChIMES-LSQ](https://github.com/rk-lindsey/chimes_lsq).

    Only support
    1. Unary system.
    2. Two- and three-body model.
    3. Morse type transformation.
    4. Tersoff smoothing.
    """

    def __init__(self):
        return

    def tersoff_smooth(self, crds, morse_fo, crds_out):
        """
        
        Tersoff smooth function.
        
        .. math::
            f_s(r_{ij}) = 
            \\begin{cases} 
            0, & \\text{if } r_{ij} > r_{\\text{cut},out} \\
            1, & \\text{if } r_{ij} < d_t \\
            \\frac{1}{2} + \\frac{1}{2} \\sin\\left(\\pi \\left[\\frac{r_{ij} - d_t}{r_{\\text{cut},out} - d_t}\\right] + \\frac{\\pi}{2}\\right), & \\text{otherwise}
            \\end{cases}\\
            d_t = r_{\\text{cut},out} * (1-f_o)
            
        
        See https://doi.org/10.1103/PhysRevB.39.5566 and https://doi.org/10.1038/s41524-024-01497-y

        Args:
            crds (np.array): Inter-particle distances :math:`r_{ij}`.
            morse_fo (float): Smoothing factor :math:`f_o`.
            crds_out (float): Radial outer cut-off :math:`r_{\\text{cut},out}`.

        Returns:
            f_s (np.array): Tersoff smooth function :math:`f_s`.
        """
        y = np.zeros(crds.shape)
        dt = crds_out * (1 - morse_fo)

        mask_out = crds > crds_out
        mask_in = crds < dt
        mask_between = ~(mask_out | mask_in)

        y[mask_in] = 1.0

        frac = (crds[mask_between] - dt) / (crds_out - dt)
        y[mask_between] = 0.5 + 0.5 * np.sin(np.pi * frac + 0.5 * np.pi)
        return y

    def morse_trans(self, crds, crds_in, crds_out, morse_lambda):
        """
        Morse type transformation.

        .. math::
            x_{ij} = \\exp{(-r_{ij}/\\lambda_\\mathrm{Morse})}

            x_{\\text{cut},in} = \\exp{(-r_{\\text{cut},in}/\\lambda_\\mathrm{Morse})}

            x_{\\text{cut},out} = \\exp{(-r_{\\text{cut},out}/\\lambda_\\mathrm{Morse})}

        See https://doi.org/10.1021/acs.jctc.7b00867.

        Args:
            crds (np.array): Inter-particle distances :math:`r_{ij}`.
            crds_in (float): Radial inner cut-off :math:`r_{\\text{cut},in}`.
            crds_out (float): Radial outer cut-off :math:`r_{\\text{cut},out}`.
            morse_lambda (float): Transformation parameter.

        Returns:
            tuple(
                transformed Inter-particle distances (np.array) :math:`x_{ij}`,

                transformed inner cut-off (float) :math:`x_{\\text{cut},in}`,

                transformed outer cut-off (float) :math:`x_{\\text{cut},out}`
            )
        """
        x = np.exp(-crds / morse_lambda)
        x_in = np.exp(-crds_in / morse_lambda)
        x_out = np.exp(-crds_out / morse_lambda)
        return x, x_in, x_out

    def rescale_into_s(self, x, x_in, x_out):
        """
        Helper function to rescale the coordinates into the interval [-1, 1], where the Chebyshev polynomials are defined.

        Args:
            x (np.array): Coordinates :math:`x_{ij}`.
            x_in (float): Coordinates inner cut-off :math:`x_{\\text{cut},in}`.
            x_out (float): Coordinates outer cut-off :math:`x_{\\text{cut},out}`.

        Returns:
            s (np.array): Rescaled coordinates :math:`s_{ij}`.
        """
        x_avg = (x_in + x_out) / 2
        x_diff = np.abs(x_in - x_out) / 2
        s = (x - x_avg) / x_diff
        return s

    def make_Amatrix(self, s, O2b, smooth_f, N_particles):
        """
        Helper function that produces two-body design matrix A.
        
        See https://doi.org/10.1021/acs.jctc.7b00867.

        Args:
            s (np.array): Rescaled coordinates.
            O2b (int): Maximum Chebyshev polynomial order.
            smooth_f (np.array): Smoothing function.
            N_particles (int): Number of particles in a system.

        Returns:
            A (np.array): The design matrix, having the dimension (n_dimers \\times n_polynomial_order + 1)\\
                The last column represents the number of particles in a system.
        """
        n_datapoints = s.shape[0]
        assert n_datapoints == smooth_f.shape[0], "number of data points does not match"

        A = []
        for o in range(1, O2b + 1):
            column = eval_chebyt(o, s) * smooth_f
            A.append(column)
        A.append(np.ones_like(s) * N_particles)

        return np.column_stack(A)

    def make_3bAmatrix(
        self,
        s_ij,
        s_ik,
        s_jk,
        smooth_f_ij,
        smooth_f_ik,
        smooth_f_jk,
        O2b,
        O3b,
        N_particles,
    ):
        """
        Helper function that produces two- plus three-body design matrix A.
        
        See https://doi.org/10.1021/acs.jctc.7b00867.
        
        Note: scipy.special.eval_chebyt that used to calculate Chebyshev polynomials 
        can still return extrapolation value when s is outside the interval of [-1, 1]. 
        As a result, the smoothing function should guarantee that the extrapolation 
        value is zero, so it doesn't influence the A matrix generation.

        Args:
            s_ij (np.array): Rescaled coordinates between particle i and j.
            s_ik (np.array): Rescaled coordinates between particle i and k.
            s_jk (np.array): Rescaled coordinates between particle j and k.
            smooth_f_ij (np.array): Smoothing function between particle i and j.
            smooth_f_ik (np.array): Smoothing function between particle i and k.
            smooth_f_jk (np.array): Smoothing function between particle j and k.
            O2b (int): Maximum two-body Chebyshev polynomial order.
            O3b (int): Maximum three-body Chebyshev polynomial order (product of three polynomials).
            N_particles (int): Number of particles in a system.

        Returns:
            A (np.array): The design matrix, having the dimension (n_configurations, valid_n_polynomial_order + 1)\\
                The last column represents the number of particles in a system.
        """
        n_datapoints = s_ij.shape[0]
        assert s_ik.shape[0] == n_datapoints, "number of data points does not match"
        assert s_jk.shape[0] == n_datapoints, "number of data points does not match"
        assert (
            smooth_f_ij.shape[0] == n_datapoints
        ), "number of data points does not match"
        assert (
            smooth_f_ik.shape[0] == n_datapoints
        ), "number of data points does not match"
        assert (
            smooth_f_jk.shape[0] == n_datapoints
        ), "number of data points does not match"

        print_out_list = []
        A = []

        print_out_list.append("Two-body order:")
        for o in range(1, O2b + 1):
            print_out_list.append(str(o))
            column = (
                eval_chebyt(o, s_ij) * smooth_f_ij
                + eval_chebyt(o, s_jk) * smooth_f_jk
                + eval_chebyt(o, s_ik) * smooth_f_ik
            )
            A.append(column)

        print_out_list.append("Three-body order & equivalent terms:")
        count_string_list = [str(i) for i in range(0, O3b)]
        desire_order = "".join(count_string_list)

        combinations = list(itertools.combinations_with_replacement(desire_order, 3))[
            O3b:
        ]

        for irr_combination in combinations:
            possible_terms = list(
                set(
                    list(
                        itertools.permutations(
                            irr_combination[0]
                            + irr_combination[1]
                            + irr_combination[2],
                            3,
                        )
                    )
                )
            )
            column = 0
            print_out_list.append(
                str(possible_terms[0]) + " " + str(len(possible_terms))
            )
            for term in possible_terms:
                n1 = float(term[0])
                n2 = float(term[1])
                n3 = float(term[2])
                column += (
                    smooth_f_ij
                    * smooth_f_ik
                    * smooth_f_jk
                    * eval_chebyt(n1, s_ij)
                    * eval_chebyt(n2, s_ik)
                    * eval_chebyt(n3, s_jk)
                )
            A.append(column)

        A.append(np.ones_like(s_ij) * N_particles)

        for item in print_out_list:
            print(item)
        return np.column_stack(A)

    def solve_LSQ_SVD(
        self,
        A,
        b,
        svd_regularization_ratio=1e-5,
        normal_eq=False,
        if_return_svd_results=False,
    ):
        """
        Solve ordinary least square problem through TSVD and return the solution vector :math:`c`.

        Loss function:
        .. math::
            \\mathcal{L} = ||Ac-b||^2

        Args:
            A (np.array): Configuration traning data with dimensions (n_configurations, valid_n_polynomial_order + 1).
            b (np.array): Energy labeling data with dimensions (n_configurations,).
            svd_regularization_ratio (flaot): TSVD regularization strength. Drop the pricipal
                componenets and sigular values if the corresponfing singular values
                samller than the maximum singular value * svd_regularization_ratio.
                Defaults to :math: `10^{-5}`.
            if_return_svd_results (bool): If retrun left and right singular vectors.
                Defaults to False.

        Returns:
            c (np.array): Polynomial coefficients with dimension (n_polynomials,).

        """
        assert (
            A.shape[0] == b.shape[0]
        ), "A and b must have the same row dimension (data points number)"

        if normal_eq:
            ATA = A.T @ A
            ATb = A.T @ b

            U, sigma, VT = np.linalg.svd(ATA, full_matrices=False)
            drop_idx = (
                sigma / sigma[0] < svd_regularization_ratio * svd_regularization_ratio
            )
            inv_sigma = 1 / sigma
            inv_sigma[drop_idx] = 0.0
            c = VT.T @ np.diag(inv_sigma) @ U.T @ ATb
        else:
            U, sigma, VT = np.linalg.svd(A, full_matrices=False)
            drop_idx = sigma / sigma[0] < svd_regularization_ratio
            inv_sigma = 1 / sigma
            inv_sigma[drop_idx] = 0.0
            c = VT.T @ np.diag(inv_sigma) @ U.T @ b

        if if_return_svd_results:
            return c, U, sigma, VT
        else:
            return c

    def solve_L2_LSQ(self, A, b, gamma, mode="qr"):
        """
        Solve L2 regularized least square problem and return the solution vector :math:`c`.
        Default to use QR factorization.

        Loss function:
        .. math::
            \\mathcal{L} = ||Ac-b||^2 + \\gamma *||c||^2

        Args:
            A (np.array): Configuration traning data with dimensions (n_configs, n_polynomials).
            b (np.array): Energy labeling data with dimensions (n_configurations,).
            gamma: L2 regularization strength.
            mode: Method to solve the linear system. Valid modes are "qr", "cholesky", "svd", "sklearn". Defaults to qr.

        Returns:
            c: Polynomial coefficients with dimension (n_polynomials,).
        """
        assert (
            A.shape[0] == b.shape[0]
        ), "A and b must have the same row dimension (data points number)"

        # form normal equation to add regularization
        if mode != "svd" or "sklearn":
            ATA = A.T @ A
            ATb = A.T @ b
            ATA_reg = ATA
            ATA_reg.flat[
                :: A.shape[1] + 1
            ] += gamma  # add gamma along the diagoanl elements

        if mode == "qr":
            # householder QR
            Q, R = np.linalg.qr(ATA_reg)

            # multiply Q.T on both side and perform back substitution
            c = scipy.linalg.solve_triangular(R, Q.T @ ATb)
        elif mode == "cholesky":
            # cholesky factorization
            L, low = scipy.linalg.cho_factor(ATA_reg)
            # forward and backward substitution
            c = scipy.linalg.cho_solve((L, low), ATb)
        elif mode == "svd":
            U, s, VT = np.linalg.svd(A, full_matrices=False)
            keep_mask = (
                s > s.max() * 1e-15
            )  # limit the maximum 2-norm condition number to be 1e-15, according to np.linalg.pinv
            s_truncated = s[keep_mask]
            rank = s_truncated.shape[0]

            d = s_truncated / (s_truncated * s_truncated + gamma)
            UT_b = U[:, :rank].T @ b
            d_UT_b = d * UT_b
            c = VT[:rank, :].T @ d_UT_b
        elif mode == "sklearn":
            reg = linear_model.Ridge(alpha=gamma)
            reg.fit(A, b)
            c = reg.coef_
        return c

    def solve_L1_LSQ_coordinate_descent(self, A, b, gamma):
        """
        Copy from [ChIMES-LSQ](https://github.com/rk-lindsey/chimes_lsq).
        """
        reg = linear_model.Lasso(alpha=gamma, fit_intercept=False, max_iter=100000)
        reg.fit(A, b)
        return reg.coef_

    def solve_L1_LSQ_LARS(self, A, b, gamma):
        """
        Copy from [ChIMES-LSQ](https://github.com/rk-lindsey/chimes_lsq).
        """
        reg = make_pipeline(
            StandardScaler(with_mean=False, with_std=False),
            linear_model.LassoLars(
                alpha=gamma,
                fit_intercept=False,
                fit_path=False,
                verbose=True,
                max_iter=100000,
            ),
        )
        reg.fit(A, b)
        return reg.coef_.ravel()


def parse_xyzf(mb_xyzf_fn, N_particles):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pmf_data = np.genfromtxt(mb_xyzf_fn, skip_header=1, invalid_raise=False)[:, -1]
        particle_data = np.genfromtxt(mb_xyzf_fn, skip_header=2, invalid_raise=False)[
            :, 1:4
        ]

    num_frame = int(particle_data.shape[0] / N_particles)
    particle_data = particle_data.reshape(num_frame, N_particles, 3)
    return pmf_data, particle_data
