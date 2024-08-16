#### IMPORTS ###

import sympy as sp
import numpy as np
from scipy import linalg as la
from numpy import linalg as nla
from scipy.integrate import ode
from types import SimpleNamespace
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import linear_sum_assignment


#####################
#### Sympy variables ####

s0, sx, sy, sz = sp.Matrix([[1, 0], [0, 1]]), sp.Matrix([[0, 1], [1, 0]]), sp.Matrix(
    [[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])
T_p, t_sp, tp_sp, k_x, k_y, k_z = sp.symbols('T_p t_sp tp_sp k_x k_y k_z', real=True)
A_x, A_y, A_z, A0_x, A0_y, A0_z = sp.symbols('A_x A_y A_z A0_x A0_y A0_z')
sp.init_printing(use_unicode=True)


#####################
#### Some auxiliary functions ####
def sorted_eigs(ev):
    """
    Function to sort the eigenvectors according to the eigenvalue order.
    :param ev: list of [eigenvalues, eigenvectors]. This is the usual output of the diagonalizing routines.
    :return: sorted list of [eigenvalues, eigenvectors]
    """

    evals, evecs = ev
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def solve_time_evolution(Ht, ts, omega, quasienergies=False, steps=False):
    """
    Solver of the time-evolution of a given time-dependent Hamiltonian.
    :param ts: array containing the time t where to calculate U(t)
    :param omega: gives the units! Notice that Ht is adimensional
    :param quasienergies: True or False. If True it returns the phase of the U(t) eigenvalues, to obtain, for example
        the Floquet quasienergies
    :param steps: True or False. If True it returns the U(t) at each ts.
    :return: U(t). If quasienergies=True it the phase of the U(t) eigenvalues. If steps= True, return the U(t) at
        each time ts.
    """

    norbs = Ht(0).shape[0]
    psi0 = np.eye(norbs, dtype=complex)
    dt = np.diff(ts)[0]

    def dydt(t, y):  # To solve de ODE we need the units back
        return -1j * np.matmul(Ht(t) * omega, np.reshape(y, (norbs, norbs)).T).ravel('F')

    solver = ode(dydt)
    solver.set_integrator('zvode')  # zvode is the proper solver for complex numbers
    solver.set_initial_value(psi0.ravel('F'), ts[0])

    Us, tsCheck = [], []
    if steps:
        tsCheck.append(solver.t)
        Us.append(np.reshape(solver.y, (norbs, norbs)).T)

    while solver.successful() and len(tsCheck) < len(ts):
        solver.integrate(solver.t + dt, step=False)
        if steps:
            tsCheck.append(solver.t)
            Us.append(np.reshape(solver.y, (norbs, norbs)).T)
    if not steps:
        U = np.reshape(solver.y, (norbs, norbs)).T
        if quasienergies:
            phases = np.angle(la.eigvals(U))
            energies = np.sort(-phases / (2 * np.pi))
            return U, energies
        else:
            return U
    else:
        return np.array(Us)


################
#### Hamiltonian classes
class Hamiltonian:
    """
    A class to define H(t) via the usual Peierls substitution of a vector potential A = (A_x(t), A_y(t), A_z(t)) in a
    momentum-dependent Hamiltonian H_0(k_x, k_y, k_z), such that
        H (k_x, k_y, k_z, t) = H_0(k_x + A_x(t), k_y + A_y(t), k_z + A_z(t)).

    Attributes
    ---------------
    h0_k : H_0 time-independent Hamiltonian. In SymPy form with a dependence on k_x, k_y, k_z.
    Axt, Ayt, Azt : A_x(t), A_y(t), A_z(t) components of the vector potential. Symbolic expression.
        The symbol for time in the SymPy expressions is t_sp. So, for example: sp.sin(2 * sp.pi / T_p * t_sp) is the
        usual $sin(2\pi t/T)$. The symbol for the period T employed in the time-dependent part is T_p. So, the integral
        for the Fourier expansion is for t_sp \in (0, T_p).
    omega : frequency, defined as 2 pi /T_p, where T_p is the period.
    h_timedep : H(t) in Sympy form.
    norbs : number of orbitals of H_0
    ham_dict : dictionary with the symbolic variables of the Hamiltonian expression.

    Methods
    -------
    fourier_elements(n) : calculates the Fourier expansion of the time-dependent Hamiltonian. In symbolic form, up to
        n-th harmonic order.
    fourier_elements_lambify (n, *par_var, par_fix={}) : evaluates the Fourier expansion up to the n-th harmonic order
        in a NumPy function with variables *par_var. The variables in par_fix are already replaced by their numerical
        value.
    time_evolutionU (params, ts, omega) : calculates the time evolution operator U(t) for a given parameter set.

    """

    def __init__(self, h0_k, Axt=0, Ayt=0, Azt=0, ham_symbols=None):
        """
        Parameters
        ----------
        :param h0_k: H_0(k_x, k_y, k_z) time-independent Hamiltonian in a SymPy form.
        :param Axt: A_x(t) x-component of the time-dependent vector potential.
        :param Ayt: A_y(t) y-component of the time-dependent vector potential.
        :param Azt: A_z(t) z-component of the time-dependent vector potential.
        :param ham_symbols: dictionary of SymPy symbols employed in the definition of H_0.
        """
        if ham_symbols is None:
            ham_symbols = {}

        self.h0_k = h0_k
        self.Axt = Axt
        self.Ayt = Ayt
        self.Azt = Azt
        self.omega = 2 * sp.pi / T_p
        self.h_timedep = h0_k.subs([(k_x, k_x + Axt), (k_y, k_y + Ayt), (k_z, k_z + Azt)])
        self.norbs = int(h0_k.shape[0])
        self.ham_dict = dict(**{str(b): b for b in ham_symbols},
                             **{str(i): i for i in [T_p, t_sp, k_x, k_y, k_z, A_x, A_y, A_z, tp_sp]})

    # locals().update(ham_dict)

    def fourier_elements(self, n):
        """
        Method that calculates the Fourier-Floquet expansions for the n-th harmonic order given by:
             \frac{1}{T} \int_T dt H(t) e^{i(n)\omega t}
        :param n: order for the Fourier expansions.
        :return: SymPy expression with of the evaluated integral for the Fourier expansion.
        """
        return sp.simplify(
            1 / T_p * sp.integrate(self.h_timedep * sp.exp(sp.I * n * self.omega * t_sp), (t_sp, 0, T_p)))

    def fourier_elements_lambify(self, n, *par_var, par_fix=None):
        """
        Evaluates the Fourier-Floquet of the n-th harmonic order given by the method fourier_elements( n) in a Numpy
        function with variables names given by *par_var and fixed parameters given by the input par_fix dictionary.
        :param n: order for the Fourier expansions.
        :param par_var: variables of the Numpy function. Dictionary type.
        :param par_fix: variables in the symbolic expression which are evaluated at some numerical value. Dictionary type.
        :return: NumPy expression with of the evaluated integral for the Fourier expansion.
        """
        if par_fix is None:
            par_fix = {}
        return sp.lambdify(dict(*par_var).keys(),
                           (self.fourier_elements(n)).evalf(
                               subs={self.ham_dict[key]: val for key, val in par_fix.items()}),
                           modules=["numpy"])

    def time_evolutionU(self, params, ts, omega, quasienergies=False, steps=False):
        """
        Calculates the time-evolution operator U(t) for the time-dependent H(t).
        :param params: dictionary containing the parameters of H(t) to be substituted with numerical values.
        :param ts: array of time to calculate U(t).
        :param omega: frequency. Gives the units! Notice that H(t) is adimensional.
        :param quasienergies: True or False. If True extract the phase of the U(t) eigenvalues.
        :param steps: if True is returning all the U(t). Default value: False.
        :return: U(t) for the final t. If quasienergies == True it returns U(t) and the phases of the U(t) eigenvalues.
        If steps ==  True, returns U(t) for all the t \in ts.
        """
        Ht = sp.lambdify(t_sp, self.h_timedep.evalf(subs={self.ham_dict[key]: val for key, val in params.items()}),
                         modules=["numpy"])
        return solve_time_evolution(Ht, ts, omega, quasienergies, steps)


class Hamiltonian_FloquetFourier:
    """
    A class to define the Floquet-Fourier Hamiltonian H^F_{mn} in a matrix form.
    The Floquet-Fourier Hamiltonian is defined by:
         H^F_{mn} = \frac{1}{T} \int_T dt H(t) e^{i(m-n)\omega t} - m \hbar \omega \delta_{m,n}~,
    The SymPy integrals are already evaluated and H^F is adimensional (in units of \hbar \omega).

    Attributes
    ---------------
    fourier_matrices : List of the Floquet-Fourier expansion up to order Nfourier. List of NumPy functions.
    N : Number of replicas considered. The range of the replicas index is given by Nrange = (-N, -N+1, ..., N-1, N).
    Nfourier : order of the highest Fourier expansion element.
    Nrange : Range of the replicas index.
    norbs : number of orbitals of H_0.
    ham0 : H_0(k_x,k_y,k_z) in Numpy function form.


    Methods
    -------
    fourier_hamiltonian : calculates the Floquet-Fourier Hamiltonian in matrix form evaluating all the parameters in the
        dictionary params.
    fourier_spectrum : calculates the Fourier-Floquet spectrum.

    """

    def __init__(self, hamiltonian, *par_var, par_fix=None, N=10, Nfourier=5):
        """
        Parameters
        ----------
        :param hamiltonian: H(t) Hamiltonian. An element of the class `Hamiltonian'.
        :param par_var: parameters to keep as variables such that H^F(*par_var). Dictionary type.
        :param par_fix: paramaters to fix to a given numerical value. Dictionary type.
        :param N: Number of replicas to be considered.
        :param Nfourier: Number of the highest Fourier expansion element to be considered.
        """
        if par_fix is None:
            par_fix = {}
        self.fourier_matrices = [hamiltonian.fourier_elements_lambify(n, *par_var, par_fix=par_fix) for n in
                                 range(Nfourier)]  ## This is a list of Fourier functions
        self.N = N
        self.Nfourier = Nfourier
        self.Nrange = np.arange(-N, N + 1)
        self.norbs = hamiltonian.norbs
        self.ham0 = sp.lambdify(dict(*par_var).keys(),
                                (hamiltonian.h0_k).evalf(
                                    subs={hamiltonian.ham_dict[key]: val for key, val in par_fix.items()}),
                                modules=["numpy"])

    def fourier_hamiltonian(self, params):
        """
        Method for computing the matrix form of the Floquet-Fourier Hamiltonian. All free parameters have to be fixed
        in params. The matrix is adimensional, in units of \hbar \omega.
        :param params: parameters of the system fixed here for the numerical evaluation of the H^F.
        :return: Matrix of H^F_{mn}.
        """

        hamM = np.zeros((self.norbs * (2 * self.N + 1), self.norbs * (2 * self.N + 1)),
                        dtype=complex)  # initialize the Matrix
        for di in range(1, self.Nfourier):
            hamM += np.kron(np.diag(np.ones(2 * self.N + 1 - abs(di)), k=- di),  ## -d out of diagonal elements
                            self.fourier_matrices[di](**params))

        hamM += np.conjugate(np.transpose(hamM))
        hamM += np.diag(
            np.ravel([(-n + 0j) * np.ones(self.norbs) for n in self.Nrange]))  # n\hbar\omega in the diagonal

        hamM += np.kron(np.eye(2 * self.N + 1), self.fourier_matrices[0](**params))  # Diagonal elements
        return hamM

    def fourier_spectrum(self, params, states=False, allstates=False):
        """
        Diagonalizes H^F for a fixed set of parameters.
        :param params: parameters of H^F. Dictionary type.
        :param states: if True it returns the eigenvectors of the replica n = 0 -> 1FBZ.
        :param allstates: if True it returns all the eigenvectors of the Matrix.
        :return: Eigenvalues of H^F.
            If states == True, it returns ens, ens_1FBZ, vecs_1FBZ.
            If allstates == True, it return ens, vecs.
            The eigenvectors vecs follow the usual structure-> vecs[:,i] corresponds to the i-th eigenvalue.
        """

        if states:
            # Note that we can not use sparse linalg because the ordering of the states will be a mess in
            # non-particle-hole sym Hamiltonians
            e0, wfs0 = sorted_eigs(nla.eigh(self.fourier_hamiltonian(params)))
            if allstates:
                return e0, wfs0
            else:
                return (e0, e0[self.norbs * self.N: self.norbs * (self.N + 1)],
                        np.array(wfs0[:, self.norbs * self.N: self.norbs * (self.N + 1)]))
        else:
            return nla.eigvalsh(self.fourier_hamiltonian(params))


class ObservablesFF:
    """
    Auxiliary class to compute some basic Floquet observables such as the photoelectron spectroscopy amplitude:
          P(\Omega) = \sum_{b, m} |f_b|^2 |f_{m,b}|^2 ~ \delta (xi_b +m\hbar\omega-\hbar\Omega)~,
    where f_{m,b} = < u_{b}^{(m)} | psi(0) >  is defined as the projection of the initial state
    over the m-th replica.
    The time-averaged DOS of states is also calculated by:
         \bar{\rho}(E) = \sum_{b,m} A_b^{(m)} \delta (xi_b +m \hbar \omega-E)~,
    where A_b^{(m)} = <u_b^{(m)}|u_b^{(m)}> are the Fourier components of the Floquet function.

    Attributes
    ---------------
    N : Number of replicas considered
    momenta_name = var_name : name of the momentum considered 'k_i', in Str format.
    momenta = var_values : vales of k_i considered for the numerical diagonalization.
    ens : Fourier-Floquet spectrum as a function of the momenta 'k_i'.
    ensF, wfsF : Floquet quasienergies and corresponding wavefunctions for the replica m = 0.
    norbs : number of orbitals of H_0
    ham0_k : H_0(k_i) in Numpy function form as a function of the momentum k_i and with the rest of parameters evaluated

    Methods
    -------
    timeAveragedDOS : calculates and plots the time-averaged DOS \bar{\rho} (E) vs k_i.
    photoelAmp : calculates the photoelectron amplitude P (\Omega) vs k_i.
    photoelAmpPlot : plots the photoelectron amplitude P (\Omega) vs k_i.
    """

    def __init__(self, hamFF, var_name, var_values, params):
        """
        Parameters
        ----------
        :param hamFF: the input is a Hamiltonian_FloquetFourier.
        :param params: parameters for the variables. Dictionary type.
        :param var_name: name of the variable that is plotted in the x axis, typically a momentum. In Str type.
        :param var_values: values of the variable depicted in x-axis. Array type.
        """

        self.N = hamFF.N
        self.momenta = var_values
        self.momenta_name = var_name
        ens, ensF, wfsF = [], [], []
        for vari in var_values:
            ei, eFi, wi = hamFF.fourier_spectrum({var_name: vari, **params}, states=True)
            ens.append(ei)
            ensF.append(eFi)
            wfsF.append(wi)
        self.ens, self.ensF, self.wfsF = np.array(ens), np.array(ensF), np.array(wfsF)
        self.norbs = hamFF.norbs
        self.ham0_k = lambda x: hamFF.ham0(**{var_name: x}, **params)  ## H_0(k_i) function of k_i only

    def timeAveragedDOS(self, ax=None, Nmax=3, cbar=False, fig=None, axs=None,
                        cmap=mpl.colormaps['inferno_r']):
        """
        Plots the time-averaged DOS $\bar{\rho} (E)$ vs 'k_i'.
        :param ax: (matplotlib.axes.Axes instance or None) – If ax is None no plot is created.
        :param Nmax: Maximum number of replicas plotted.
        :param cbar: True or False. If True, it includes a colorbar for the figure.
        :param fig: matplotlib figure or None. Include the figure in case of a multiple axes figure.
        :param axs: list of matplotlib.axes.Axes. Include the axs in case of a multiple axes figure.
        :param cmap: matplotlib colormap employed in the plot.
        :return: array containing the time-averaged DOS values (vs 'k_i'). The indices of the array are [momentum, band].
        """
        plotrange = np.arange(self.N - Nmax - 1, self.N + Nmax + 1)
        norm = mpl.colors.PowerNorm(vmin=0, vmax=1, gamma=0.5)
        for i in range(self.norbs):
            wfi = self.wfsF[:, :, i]
            spectral = np.sum(np.abs(wfi.reshape(len(self.momenta), (2 * self.N + 1), self.norbs)) ** 2,
                              -1)  # sum over axis -1 to get |<psi|psi>|**2
            if ax is not None:
                z1 = [ax.scatter(self.momenta, self.ensF[:, i] + (j - self.N),
                                 c=spectral[:, j], cmap=cmap, norm=norm, s=4) for j in plotrange]
        if ax is not None:
            ax.set_facecolor('lightgrey')
            ax.set(ylim=(-Nmax, Nmax), xlim=(self.momenta[0], self.momenta[-1]))
            ax.set(xlabel=(r'$v{}/\omega$'.format(self.momenta_name)),
                   ylabel=r'$E/(\hbar \omega)$')
            if cbar:
                if fig is None:
                    print('Include a figure in the input')
                else:
                    cbar_ax = fig.colorbar(z1[0], ax=axs, pad=0.05, shrink=0.8, aspect=15, fraction=0.05)
                    cbar_ax.ax.set_title(r'DOS', fontsize=20, y=1.1, pad=-8)
        return spectral

    def photoelAmp(self, psi0=None, fk_out=False, band_initial=0, ):
        """
        Calculates the Photoelectron Amplitude for a given initial state psi0. From the  expression:
            P(\Omega) = | \int_{-infty}^{infty} dt e^{i \Omega t} <psi(t)|psi(0)> |^2,
        or equivalently:
            P(\Omega) = \sum_{b, m} |f_b|^2 |f_{m,b}|^2  \delta ( xi_b + m \hbar \omega-\hbar \Omega),
        where f_{m,b} = <u_b^{(m)}|psi(0)>.
        The initial state can be either given as an input or it is calculates as the eigenvector of H_0 of the band
            band_initial.
        :param psi0: Array. Initial state.
        :param fk_out: If True, returns the coefficients f_{m,b} = <u_b^{(m)}|psi(0)>.
        :param band_initial: Integer. Number of band of H_0 employed as initials state. If psi0 is not None, this
            is skipped.
        :return: If fk_oyt is True it returns the coefficients f_{m,b} vs k_i, otherwise it returns the Photoelectron
            amplitude vs k_i.
        """

        fk_mb, ek_F = [], []

        for ki, kval in enumerate(self.momenta):
            if psi0 is None:
                psi00 = nla.eigh(self.ham0_k(kval))[1][:, band_initial]
            else:
                psi00 = psi0[ki]

            f_mb = []
            for vec, en in zip((self.wfsF[ki]).T, self.ensF[ki]):  # Tranpose the eigenvectors
                phiF = np.reshape(vec, (2 * self.N + 1, self.norbs))
                f_mb.append([np.sum(np.conjugate(phi) * psi00) for phi in phiF])
            fk_mb.append(f_mb)
        if fk_out:
            return np.array(fk_mb)
        else:
            fk_mb = np.array(fk_mb)  # indices = momentum, band, Fourier mode
            f_b2 = np.sum(np.abs(fk_mb) ** 2, axis=2)  # indices = momentum, band
            pk = np.array([[np.abs(fk_mb[k, b]) ** 2 * f_b2[k, b] for b in range(self.norbs)]
                           for k in range(len(self.momenta))])  # p = sum_{b,m} |f_b|^2 |f_mb|^2
            return pk

    def photoelAmpPlot(self, ax, Nmax=3, psi0=None, band_initial=0, cmap=mpl.colormaps['inferno_r'], vmax=0.25,
                       gamma=0.5, cbar=False, fig=None, axs=None):
        """
        Plots (and calculates) the photo-electron amplitude P(\Omega).
        :param ax: matplotlib.axes.Axes instance where the plot is done.
        :param Nmax: Maximum number of replicas plotted.
        :param psi0: Array. Initial state.
        :param band_initial: Integer. Number of band of H_0 employed as initials state. Default value is valence band
            b = 0. If psi0 is not None, this is skipped.
        :param cmap: matplotlib colormap employed in the plot.
        :param vmax: vmax of the colormap.
        :param gamma: gamma of the colormap PowerNorm.
        :param cbar: True or False. If True, it includes a colorbar for the figure.
        :param fig: matplotlib figure or None. Include the figure in case of a multiple axes figure.
        :param axs: list of matplotlib.axes.Axes. Include the axs in case of a multiple axes figure.
        :return: figure with the photo-electron amplitude plotted. If cbar: returns figure and cbar axis.
        """
        norm = mpl.colors.PowerNorm(vmin=0, vmax=vmax, gamma=gamma)
        plotrange = np.arange(self.N - Nmax, self.N + Nmax + 1)
        ps = self.photoelAmp(psi0=psi0, band_initial=band_initial)
        z1 = [[ax.scatter(self.momenta, self.ensF[:, b] + (i - self.N), c=ps[:, b, i], cmap=cmap, norm=norm, s=4)
               for i in plotrange] for b in range(self.norbs)][0]
        ax.set(facecolor='lightgrey', ylim=(-Nmax, Nmax), xlabel=(r'$v{}/\omega$'.format(self.momenta_name)),
               ylabel=(r'$E/(\hbar \omega)$'))
        if cbar:
            cbar_ax = fig.colorbar(z1[0], ax=axs, pad=0.03, shrink=0.8, aspect=15, fraction=0.05, extend='max')
            cbar_ax.ax.set_title(r'P(E)', fontsize=20, y=1.1, pad=-14)
            return fig, cbar_ax
        else:
            return fig


class Hamiltonian_ttp:
    """
    A class to define the Hamiltonian in t-t' formalism H (k_x, k_y, k_z, a(t), t).
    The Hamiltonian is built from the H_0 via the Peierls substitution:
         H (k_x, k_y, k_z, t) = H_0(k_x + A_x(t), k_y + A_y(t), k_z + A_z(t)).
    However, in this case, the vector potential are factorizes as:
         A_i (t) = A_i^{env} (t) V( t) ,
    where i = x,y,z. The envelop part is A_i^{env} (t) while V(t) = V(t+T) is periodic.

    Attributes
    ---------------
    h0_k : H_0(k_x,k_y,k_z) in Sympy form.
    norbs : number of orbitals of H_0.
    N : Number of replicas considered. The range of the replicas index is given by Nrange = (-N, -N+1, ..., N-1, N).
    Vxt, Vyt, Vzt : Sympy expression of the time-periodic part of the vector potential: A_i (t) = A_i^{env}(t) * V_i(t).
        V_i(t) = V_i(t+T)
    par_fix: variables in the symbolic expression which are evaluated at some numerical value. Dictionary type.
    omega : frequency, defined as 2 pi /T_p, where T_p is the period. Symbolic variable.
    omegaev : numerical value of the frequency.
    h_timedep : H(t) in Sympy form.
    ham_dict : dictionary with the symbolic variables of the Hamiltonian expression.
    Axenv, Ayenv, Azenv : A^{env}_x(t), A^{env}_y(t), A^{env}_z(t) components of the aperiodic envelop of the vector
        potential. Symbolic expression.
    ham_Adep : Symbolic expression of the t-t' Hamiltonian at fixed envelop amplitude A0_i. It is obtained by:
        $ H (A0_x, A0_y, A0_z, k_x, k_y, k_z, t) = H_0(k_x + A0_x * V_x(t), k_y +A0_y * V_y(t), k_z + A0_z * V_z(t) ) $
    ham_AdepFF : Floquet-Fourier Hamiltonian as a function of the envelope amplitudes A0_x, A0_y and A0_z.

    Methods
    -------
    time_evolutionU : Calculates the time-evolution of H(t).
    ifs_basis_flip : Calculates the instantaneous Floquet basis for each fixed A0_i and tracks the bands flipping.
    ifs_Chamilt : Calculates the Hamiltonian for the evolution of the c_alpha coefficients:
         i \hbar \frac{d c_{\alpha}}{dt} = \sum_{\beta} H ^{tt'}_{\alpha \beta}(a(t))~c_\beta(t)~,
        where a(t) is, in the previous notation a(t) = (Axevn(t), Ayenv(t), Azenv(t)).
    """

    def __init__(self, h0_k, par_var_ham, par_fix=None, Vxt=0, Vyt=0, Vzt=0,
                 Axenv=0 * t_sp, Ayenv=0 * t_sp, Azenv=0 * t_sp, N=10, Nfourier=5, ham_symbols=None):
        """
        Parameters
        ----------
        :param h0_k: H_0(k_x, k_y, k_z) time-independent Hamiltonian in a SymPy form. Sympy expression.
        :param par_var: variables of the Numpy function. Dictionary type.
        :param par_fix: variables in the symbolic expression which are evaluated at some numerical value. Dictionary type.
        :param Vxt: V_x(t) x-component of the time-periodic part of the vector potential. Sympy expression.
        :param Vyt: V_y(t) y-component of the time-periodic part of the vector potential. Sympy expression.
        :param Vzt: V_z(t) z-component of the time-periodic part of the vector potential. Sympy expression.
        :param Axenv: A^{env}_x(t) x-component of the envelop (aperiodic) part of the vector potential. Sympy expression.
        :param Ayenv: A^{env}_y(t) y-component of the envelop (aperiodic) part of the vector potential. Sympy expression.
        :param Azenv: A^{env}_z(t) z-component of the envelop (aperiodic) part of the vector potential. Sympy expression.
        :param N: Number of replicas to be considered. Integer.
        :param Nfourier: Number of the highest Fourier expansion element to be considered. Integer.
        :param ham_symbols: dictionary of extra SymPy symbols employed in the definition of H_0. Dictionary type.
        """

        if par_fix is None:
            par_fix = {}
        if ham_symbols is None:
            ham_symbols = {}
        self.h0_k = h0_k
        self.norbs = int(h0_k.shape[0])
        self.N = N
        self.Vxt = Vxt
        self.Vyt = Vyt
        self.Vzt = Vzt
        self.par_fix = par_fix
        self.omega = 2 * sp.pi / T_p
        self.omegaev = 2 * np.pi / par_fix['T_p']
        self.h_timedep = h0_k.subs([(k_x, k_x + Axenv * Vxt), (k_y, k_y + Ayenv * Vyt), (k_z, k_z + Azenv * Vzt)])
        self.ham_dict = dict(**{str(b): b for b in ham_symbols},
                             **{str(i): i for i in [T_p, t_sp, k_x, k_y, k_z,
                                                    A_x, A_y, A_z, tp_sp,
                                                    A0_x, A0_y, A0_z]})
        self.Axenv = Axenv.evalf(subs={self.ham_dict[key]: val
                                       for key, val in par_fix.items()})
        self.Ayenv = Ayenv.evalf(subs={self.ham_dict[key]: val
                                       for key, val in par_fix.items()})
        self.Azenv = Azenv.evalf(subs={self.ham_dict[key]: val
                                       for key, val in par_fix.items()})
        self.ham_Adep = Hamiltonian(self.h0_k, Axt=A0_x * Vxt, Ayt=A0_y * Vyt, Azt=A0_z * Vzt, ham_symbols=ham_symbols)
        self.ham_AdepFF = Hamiltonian_FloquetFourier(self.ham_Adep, dict(A0_x=None, A0_y=None, A0_z=None,
                                                                         **par_var_ham), par_fix=par_fix, N=N,
                                                     Nfourier=Nfourier)

    def time_evolutionU(self, params, ts, quasienergies=False, steps=False):
        """
        Calculates the time-evolution of H(t). Has the same inputs and outputs of 'solve_time_evolution'.
        :param params: Dictionary of parameters to be fixed for the numerical evaluation of the Hamiltonian.
        :param ts: array containing the time t where to calculate U(t)
        :param quasienergies: True or False. If True it returns the phase of the U(t) eigenvalues, to obtain, for
            example the Floquet quasienergies
        :param steps: True or False. If True it returns the U(t) at each ts.
        :return: U(t). If quasienergies=True it the phase of the U(t) eigenvalues. If steps= True, return the U(t) at
            each time ts.
        """

        ham_t = sp.lambdify(t_sp, self.h_timedep.evalf(subs={self.ham_dict[key]: val for key, val in params.items()}),
                            modules=["numpy"])
        return solve_time_evolution(ham_t, ts, self.omegaev, quasienergies, steps)

    def ifs_basis_flip(self, ts, params_env, params_ham, tol_parallel= 0.02):
        """
        Calculates the instantaneous Floquet basis for each fixed A0 value of the envelop amplitude.
        It checks also the case flipping bands.
        :param ts: Array containing the time t where to calculate the evolution.
        :param params_env: params of the envelope of the pulse. A^{env}_i (t). Dictionary type.
        :param params_ham: params of the Hamiltonian. Dictionary type.
        :param tol_parallel: tolerance to check the parallel transport condition
        :return: List of Floquet-Fourier energies eL, eigenvectors wL and amplitudes of the envelope Alist.
            The indices are: eL[envelope amplitude, band], wL[envelope amplitude, orbitals and sites, band].
            Alist is a len(ts)x3 Numpy array with [A_x, A_y, A_z] for each time.
        """

        def best_match(psi1, psi2, threshold=None):
            """
            Find the best match of two sets of eigenvectors.
            Code adapted from https://quantumtinkerer.tudelft.nl/blog/connecting-the-dots/

            Parameters:
            -----------
            psi1, psi2 : numpy 2D complex arrays
                Arrays of initial and final eigenvectors.
            threshold : float, optional
                Minimal overlap when the eigenvectors are considered belonging to the same band.
                The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.

            Returns:
            --------
            sorting : numpy 1D integer array
                Permutation to apply to ``psi2`` to make the optimal match.
            disconnects : numpy 1D bool array
                The levels with overlap below the ``threshold`` that should be considered disconnected.
            """
            if threshold is None:
                threshold = (2 * psi1.shape[0]) ** -0.25
            Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
            orig, perm = linear_sum_assignment(-Q)
            return perm, Q[orig, perm] < threshold

        Axfunc = (
            sp.lambdify(t_sp, (self.Axenv).evalf(subs={self.ham_dict[key]: val for key, val in params_env.items()}),
                        modules=["numpy"]) if params_env['A_x'] != 0 else lambda t: 0 * t)
        Ayfunc = (
            sp.lambdify(t_sp, (self.Ayenv).evalf(subs={self.ham_dict[key]: val for key, val in params_env.items()}),
                        modules=["numpy"]) if params_env['A_y'] != 0 else lambda t: 0 * t)
        Azfunc = (
            sp.lambdify(t_sp, (self.Azenv).evalf(subs={self.ham_dict[key]: val for key, val in params_env.items()}),
                        modules=["numpy"]) if params_env['A_z'] != 0 else lambda t: 0 * t)
        Alist = np.array([Axfunc(ts), Ayfunc(ts), Azfunc(ts)]).T

        eL, wL = [], []
        for i, As in enumerate(Alist):
            par_var_subs = {'A0_x': As[0], 'A0_y': As[1], 'A0_z': As[2], **params_ham}
            ei, wi = (self.ham_AdepFF).fourier_spectrum(dict(**par_var_subs), states=True, allstates=True)
            if i == 0:
                e0, w0 = ei, wi
            else:
                perm, line_breaks = best_match(w0, wi)
                ei = ei[perm]
                intermediate = (e0 + ei) / 2
                intermediate[line_breaks] = None
                w0 = wi[:, perm]
                e0 = ei
            eL.append(e0)
            wL.append(w0[:, self.norbs * self.N: self.norbs * (self.N + 1)])
        eL = np.array(eL)
        wL = np.array(wL)
        ## Parallel transport: correct the phase
        for i in range(1, len(Alist)):
            for n in range(self.norbs):
                wL[i, :, n] /= np.sum(np.conjugate(wL[i - 1, :, n]) * wL[i, :, n], axis=0)
                wL[i, :, n] /= la.norm(wL[i, :, n])
        ## Check parallel transport
        check_parallel = np.abs(np.sum(np.diff(np.conjugate(wL), axis=0) * wL[:-1], axis=1))
        if np.max(check_parallel) > tol_parallel:
            print('Warning: problems in time discretization, parallel transport not fulfilled with '
                  'tol = {}'.format(tol_parallel))
        return eL, wL, Alist

    def ifs_Chamilt(self, ts, params_env, params_ham, Nmax=None, ifs_basis=False, tol_parallel = 0.02):

        """
        Calculates the Hamiltonian for the evolution of c_alpha according to:
            i \hbar \frac{ d c_\alpha }{dt} = \sum_\beta H ^{tt'}_{\alpha \beta}( a(t) ) c_\beta(t),
        where a(t) is, in the previous notation a(t) = (Axevn(t), Ayenv(t), Azenv(t)).
        We are using the following convention for the derivative in the envelope:
            dA = sqrt[ (dA_x)^2 + (dA_y)^2 + (dA_z)^2 ] * sign(dA_x + dA_y + dA_z)
        This works for linear and circular polarization, however the definition of this should be revisited for pulses
        with, for example, different derivative sign on each component; sign(dA_x) \neq sign(dA_y), for example.

        :param ts: Array containing the time t where to calculate the evolution.
        :param params_env: params of the envelope of the pulse. A^{env}_i (t). Dictionary type.
        :param params_ham: params of the Hamiltonian. Dictionary type.
        :param Nmax: maximum number of replicas considered in the calculation. Integer type.
        :param ifs_basis: if True it returns also the instantaneous Floquet basis data.
        :return: hamHL, lms: Hamiltonian list and labels of the replicas. If ifs_basis returns (hamHL, lms, eL, wL,
            Alist, indexCbase), so also the data of the instantaneous Floquet basis such as the quasienergies eL, the
            basis vectors wL, and the list of vector potential amplitudes Alist.
        """

        if Nmax is None:
            Nmax = int(self.N / 4)
        eL, wL, Alist = self.ifs_basis_flip(ts, params_env, params_ham, tol_parallel)
        # Index of the starting basis vs final reduced basis.
        indexCbase = np.arange(- self.norbs * Nmax, + self.norbs * (Nmax + 1)) + self.norbs * self.N
        # lms is the relative Fourier index
        lms = np.arange(- Nmax, Nmax + 1)

        # define dA, follows the definition: dA = sqrt[ (dA_x)^2 + (dA_y)^2 + (dA_z)^2 ] * sign(dA_x + dA_y + dA_z)
        diffAlist = np.diff(Alist, axis=0)
        diffA = np.sign(np.sum(diffAlist, axis=1)) * np.sqrt(np.sum(diffAlist ** 2, axis=1))  ## Numerical dA

        # Define dA/dt, it is more stable to use the derivative of the analytics expressions to avoid 0/0 below
        # numerical tolerance.
        Axfunderiv = (sp.lambdify(t_sp, sp.diff(
            (self.Axenv).evalf(subs={self.ham_dict[key]: val for key, val in params_env.items()}), t_sp),
                                  modules=["numpy"]) if params_env['A_x'] != 0 else lambda t: 0 * t)
        Ayfunderiv = (sp.lambdify(t_sp, sp.diff(
            (self.Ayenv).evalf(subs={self.ham_dict[key]: val for key, val in params_env.items()}), t_sp),
                                  modules=["numpy"]) if params_env['A_y'] != 0 else lambda t: 0 * t)
        Azfunderiv = (sp.lambdify(t_sp, sp.diff(
            (self.Azenv).evalf(subs={self.ham_dict[key]: val for key, val in params_env.items()}), t_sp),
                                  modules=["numpy"]) if params_env['A_z'] != 0 else lambda t: 0 * t)

        dummy_vec = np.array([Axfunderiv(ts), Ayfunderiv(ts), Azfunderiv(ts)])
        dAs = (np.sign(np.sum(dummy_vec, axis=0)) * np.sqrt(np.sum(dummy_vec ** 2, axis=0)))[:-1]  ## Evaluated dA/dt

        def G_nm(b1, b2, n, m):
            vec1, vec2 = wL[:, :, b1], wL[:, :, b2]

            w = np.conjugate(np.reshape(vec1, (len(diffA) + 1, 2 * self.N + 1, self.norbs)))[:-1]
            dw = np.reshape((1. / diffA * np.diff(vec2, axis=0).T).T,
                            (len(diffA), 2 * self.N + 1, self.norbs))
            # promediated value of the derivaties to avoid numerical accumulation of the unwanted phase due to the
            # numerical parallel transport condition:
            return (np.sum(np.sum(0.5 * (w[:, lms + self.N + m - n] * dw[:, lms + self.N]
                                         + w[:, lms + self.N] * dw[:, lms + self.N + n - m]), axis=2), axis=1))

        def Cham_nm(b1, b2, n, m):
            # Also here, promediated between the two Fourier-pairs indices:
            return (- 0.5j * dAs * (G_nm(b1, b2, n, m) - np.conjugate(G_nm(b2, b1, m, n))))

        sizeM = 2 * Nmax + 1
        sizeT = len(diffA)

        hamL = np.zeros((sizeT, self.norbs * sizeM, self.norbs * sizeM), dtype=np.complex128)
        hamL += np.array([np.diag(eL[ti, indexCbase] * self.omegaev) for ti in range(sizeT)])

        ## Separate the contributions inside the same replica
        for b1 in range(0, self.norbs):
            for b2 in range(b1 + 1, self.norbs):
                matorbs = np.zeros((self.norbs, self.norbs))
                matorbs[b1, b2] = 1.
                c = Cham_nm(b1, b2, 0, 0)
                hamL += np.tensordot(c, np.kron(np.eye(sizeM), matorbs), axes=0)
                # nonvectorize version: hamL += np.array([c[ti] * np.kron(np.eye(sizeM), matorbs) for ti in range(sizeT)])
        for b1 in range(self.norbs):
            for b2 in range(self.norbs):
                for ni in range(1, Nmax - 1):
                    c = Cham_nm(b1, b2, +ni, 0)
                    matorbs = np.zeros((self.norbs, self.norbs))
                    matorbs[b1, b2] = 1.
                    hamL += np.tensordot(c, np.kron(np.diag((sizeM - ni) * [1], k=ni), matorbs), axes=0)
                    # nonvectorize version: hamL += np.array([c[ti] * np.kron(np.diag((sizeM-ni)*[1], k = ni), matorbs) for ti in range(sizeT)])
        ## Built the Hamiltonian for the c_alpha evolution:
        hamHL = hamL + np.transpose(np.triu(hamL, k=1).conj(), axes=[0, 2, 1])
        if ifs_basis:
            return hamHL, lms, eL, wL, Alist, indexCbase
        else:
            return hamHL, lms


class IFS_solver:
    """
    Auxiliary class for solving the t-t' evolution in the instantaneous Floquet basis. It takes as an input a
    Hamiltonian_ttp instance and calculates the evolution of the c_alpha coefficients by the decomposition of the time-
    evolution in the instantaneous Floquet basis:
        |psi(t) > = \sum_\alpha c_\alpha (t) |u_\alpha> (a(t), t)

    Attributes
    ----------
    ChamL, lms, eL, wL, Alist, indexCbase : main outputs of Hamiltonian_ttp.ifs_Chamilt.
        ChamL : List of evaluates Hamiltonian matrices of the c_alpha evolution.
        lms : list of replicas indices (in the Fourier expansion).
        eL, wL : quasienegies and corresponding wfs for the instantaneous Floquet basis.
        Alist : len(ts)x3 list of values of the three components of the vector potential for each t \in ts.
        indexCbase : indices to convert the staring ifs basis to the results of reduced basis of the c_alpha calculation
    ts: array of times to evaluate the temporal evolution
    omegaev : numerical value of the frequency.
    N : Number of replicas considered in the Fourier basis.
    Nmax : Number of replicas considered in the building of the Hamiltonian for c_alpha.
    norbs : number of orbitals of H_0
    params : full dictionary contaning all the parameters for the envelop function and the Hamiltonian.
    ham0 : Hamiltonian matrix at t = ts[0], used as a reference for tracking the bands displacement.

    Methods
    -------
    c_t : Solves the c_alpha(t) for the given ts. Optional inputs are an initial arbitrary state psi0 or the band index
        of H0 that is employed as initial state via 'psi0band'. If psi_t is True it returns the reconstructued psi(t)
        in the spinorial basis.
    tag_fqlevels : Returns the tags in form for the (band, replica index) of the instanatenous Floquet basis employed
        for solving c_alpha(t).
    """

    def __init__(self, hamttp, ts, params_env, params_ham, Nmax = None, tol_parallel = 0.02):
        """
        Parameters
        ----------
        :param hamttp: Hamiltonian_ttp instance.
        :param ts: array of times to evaluate the temporal evolution.
        :param params_env: params of the envelope of the pulse. A^{env}_i (t). Dictionary type.
        :param params_ham: params of the Hamiltonian. Dictionary type.
        :param Nmax: maximum number of replicas considered in the calculation. Integer type.
        """

        self.ChamL, self.lms, self.eL, self.wL, self.Alist, self.indexCbase = hamttp.ifs_Chamilt( ts, params_env, params_ham, 
                                                                                     Nmax, ifs_basis=True, tol_parallel=0.02)
        self.ts = ts
        self.omegaev = hamttp.omegaev
        self.N = hamttp.N
        self.Nmax = Nmax
        self.norbs = hamttp.norbs
        self.params = dict(**params_env, **params_ham, **(hamttp.par_fix))
        self.ham0 = sp.matrix2numpy(hamttp.h_timedep.evalf(subs={hamttp.ham_dict[key]: val for key, val
                                                                 in {'t_sp': ts[0], **self.params}.items()}),
                                    dtype=np.complex128)

    def c_t(self, psi0 = None, psi0band = 0, tend = None, psi_t = True):
        """
        Solves the EDOs for the coefficients c_alpha.

        :param psi0: initial state. Array type of size norbs. If None H_0 is diagonalized and the band psi0band is
            employed as initial state.
        :param psi0band: band of H0 employed as the initial state. Integer type. By default the lowest energy band of
            H_0 is employed.
        :param tend: Ending time of the c_t integration. Float.
        :param psi_t: True or False. If True it returns also the psi (t) obtained from:
            |psi(t) > = \sum_\alpha c_\alpha (t) |u_\alpha> (a(t), t).

        :return: c_alpha (t) coefficients for t \in ts < tend. If psi_t: returns also psi(t) obtained from c_alpha.
        """
        if psi0 is None:
            e, w = nla.eigh(self.ham0)
            psi0 = w[:, psi0band] / np.linalg.norm(w[:, psi0band])
            print('State at t = {:.2f} energy = {:.4f}, wf (spinor): {}'.format(self.ts[0], e[psi0band],
                                                                                np.round(psi0, 3)))
        if tend is None:
            tend = 0.9 * self.ts[-1]

        c0bands = np.array([np.sum(np.conjugate(self.wL[0, :, nb].reshape(2 * self.N + 1, self.norbs)) * psi0,
                                   axis=1)[self.lms + self.N] for nb in range(self.norbs)])
        c0 = np.ravel([[c0bands[nb, i] for nb in range(self.norbs)] for i in range(len(self.lms))])

        dt = np.diff(self.ts)[0]

        sol = []
        def dydt(t, y):
            return -1j * np.matmul(self.ChamL[int(t // dt)], y)

        solver = ode(dydt)
        solver.set_integrator('zvode')  # zvode is the proper option to work with complex numbers
        solver.set_initial_value(c0, self.ts[0])

        while solver.successful() and solver.t < tend:
            solver.integrate(solver.t + dt, step = False)
            sol.append(solver.y)

        def psi_tfunc(ct):
            """
            Builds the psi(t) from the c_alpha coefficients:
                |psi(t) > = \sum_\alpha c_\alpha (t) |u_\alpha> (a(t), t)
            :param ct: c_alpha coefficients solved previously.
            :return: psi(t)
            """
            ts_ct = self.ts[:len(ct)]
            psitFcheck = 1j * np.zeros((len(ts_ct), self.norbs))
            ub = [np.array([np.sum(np.reshape(self.wL[ti, :, nb], (2 * self.N + 1, self.norbs))[self.lms + self.N]
                                *np.array([np.exp(-1j * self.omegaev * n * t) * np.ones(self.norbs) for n in self.lms]),
                                axis=0) for ti, t in enumerate(ts_ct)]) for nb in range(self.norbs)]
            for orb in range(self.norbs):
                for i in range(len(self.lms) * self.norbs):
                    psitFcheck[:, orb] += np.exp(1j * self.omegaev * (self.lms[i // self.norbs]) * ts_ct) * \
                                          ct[:, i] * ub[i % self.norbs][:, orb]
            return psitFcheck

        ct = np.array(sol)
        if psi_t:
            return ct, psi_tfunc(ct)
        else:
            return ct

    def tag_fqlevels(self):
        """
        Auxiliary method to return an array of size Nrange x 2 that maps the full Floquet-Fourier basis employed to the
        labels of alpha = (band, Fourier replica index).
        The replica m = 0 of the band b is defined by the limit of:
            limit (A -> 0 )  En^F_(b, m=0)  ->  En_0 (b),
        where En^F indicates the energy of the Fourier-Floquet replicas and En_0 is the eigenvalue of H_0.
        :return: Array of the same size of the Fourier-Floquet basis x 2 with the label of each element of the basis as
        alpha = (band, replica).
        """
        if max(self.Alist[0]) >= 5e-3:
            print('Error: the amplitude at t={:.3f} is not A = 0.'.format(self.ts[0]))
        else:
            e, w = nla.eigh(self.ham0)
            indmin = []
            for n in range(self.norbs):
                diffE = np.abs(self.eL[0] - e[n])
                indmin.append(np.argwhere(diffE == np.min(diffE))[0][0])
            tag_vec = np.empty((len(self.eL[0]), 2))
            for nb in range(self.norbs):
                for m in range(-self.Nmax, self.Nmax):
                    tag_vec[indmin[nb] + m * self.norbs] = [int(nb), int(m)]
            return np.nan_to_num(tag_vec)
