import numpy as np

from al_specific_components.candidate_update.candidate_updater_implementations import Stream
from basic_sl_component_interfaces import ReadOnlyPassiveLearner
from example_implementation.helpers.mapper import map_flat_output_to_shape, map_flat_input_to_shape, map_shape_input_to_flat
from helpers import X

unit_Bohr_A = 0.52917721090380


class MethanolStream(Stream):
    def __init__(self, start_coords, start_ys, pl: ReadOnlyPassiveLearner):
        start_coords = map_flat_input_to_shape(np.expand_dims(start_coords, axis=0)) / unit_Bohr_A
        initial_y = map_flat_output_to_shape(np.expand_dims(start_ys, axis=0))
        initial_eng = initial_y[0]
        initial_grad = initial_y[1]

        mass = np.array([[[12.0], [15.99491], [1.007825], [1.007825], [1.007825], [1.007825]]]) * _BatchEnsemble.unit_1u_me

        # setup ensemble
        ensemble_md = _BatchEnsemble(start_coords, mass, pl, initial_eng, initial_grad)
        ensemble_md.initialize_velocity(1000.0)

        self.trajectory_md = _VerletIntegration(ensemble_md)

    def get_element(self) -> X:
        return self.trajectory_md.propagate_timestep(1 * 1e-15 / self.trajectory_md.unit_atu_s)


class _VerletIntegration:
    unit_atu_s = 2.418884326585747e-17
    const_kB = 8.617333262e-5 / 27.21138624598853

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def _predict(self, x):
        y = map_flat_output_to_shape(self.ensemble.pl.predict_set(map_shape_input_to_flat(x))[0])

        eng, grad = y[0], y[1]
        return eng, grad

    def propagate_timestep(self, delta_t):
        # time in atu

        t = self.ensemble.traj_t[-1]
        x_t = self.ensemble.traj_x[-1]
        a_t = self.ensemble.traj_a[-1]
        v_t = self.ensemble.traj_v[-1]
        m = self.ensemble.mass
        N = self.ensemble.number_particles
        kB = self.const_kB
        sig_F = self.ensemble.sign_force

        v_t_dt_2 = v_t + 0.5 * a_t * delta_t
        x_t_dt = x_t + v_t_dt_2 * delta_t

        e_t_dt, g_t_dt = self._predict(x_t_dt)
        a_t_dt = np.squeeze(sig_F * g_t_dt / m, axis=1)

        v_t_dt = v_t_dt_2 + 0.5 * a_t_dt * delta_t
        e_kin_t_dt = np.sum(np.sum(0.5 * m * v_t_dt * v_t_dt, axis=-1), axis=-1, keepdims=True)
        p_t_dt = v_t_dt * m
        T_dt = 2 / 3 * e_kin_t_dt / N / kB

        # Add time-step
        self.ensemble.traj_x.append(x_t_dt)
        self.ensemble.traj_v.append(v_t_dt)
        self.ensemble.traj_a.append(a_t_dt)
        self.ensemble.traj_t.append(t + delta_t)
        self.ensemble.traj_F.append(sig_F * g_t_dt)
        self.ensemble.traj_E.append(e_t_dt)
        self.ensemble.traj_E_kin.append(e_kin_t_dt)
        self.ensemble.traj_T.append(T_dt)
        self.ensemble.traj_p.append(p_t_dt)

        return map_shape_input_to_flat(x_t_dt)[0]


class _BatchEnsemble:
    unit_Bohr_A = 0.52917721090380
    unit_Bohr_m = 5.2917721090380e-11
    unit_me_kg = 9.109383701528e-31
    unit_Hatree_eV = 27.21138624598853
    unit_Hatree_J = 4.359744722207185e-18
    unit_atu_s = 2.418884326585747e-17
    unit_1u_me = 1.6605390666050e-27 / 9.109383701528e-31

    const_kB = 8.617333262e-5 / 27.21138624598853
    sign_force = -1.0

    def __init__(self, coord, mass, pl, initial_eng, initial_grad, velo=None):
        """Initial settings. All properties are in atomic units.

        Args:
            coord (np.array): Initial poisition of shape (batch, N, 3) in [Bohr]
            mass (np.array): Mass of particles (bath, N, 1) in [me]
            pl (ReadOnlyPassiveLearner): passive learner for predictions
            velo (np.array): Initial velocities of shape (batch, N, 3)
        """

        # System properties
        self.mass = mass
        self.pl = pl
        self.number_particles = self.mass.shape[1]

        # Trajectory properties
        self.traj_x = []  # position in Bohr
        self.traj_v = []  # velicity in Bohr/atu
        self.traj_a = []  # acceleraton in Bohr/atu^2
        self.traj_t = []  # time in atu
        self.traj_p = []  # momentum in Bohr*me/atu
        self.traj_F = []  # force in Hatree/Bohr
        self.traj_E = []  # potential energy in Hatree
        self.traj_E_kin = []  # kinetic energy in Hatree
        self.traj_T = []  # Temperature in K

        #######################################################################

        # Set initial values i.e. traj[0]
        initial_x = coord
        if velo is None:
            initial_v = np.zeros_like(coord)
        else:
            initial_v = velo
        initial_p = initial_v * self.mass
        initial_force = self.sign_force * initial_grad
        initial_eng_kin = np.sum(np.sum(0.5 * self.mass * initial_v * initial_v, axis=-1), axis=-1, keepdims=True)
        initial_temp = 2 / 3 * initial_eng_kin / self.number_particles / self.const_kB
        initial_a = np.squeeze(initial_force / self.mass, axis=1)

        # Append 0 time step
        self.traj_x.append(initial_x)
        self.traj_v.append(initial_v)
        self.traj_E.append(initial_eng)
        self.traj_F.append(initial_force)
        self.traj_a.append(initial_a)
        self.traj_t.append(0.0)
        self.traj_E_kin.append(initial_eng_kin)
        self.traj_T.append(initial_temp)
        self.traj_p.append(initial_p)

    def initialize_velocity(self, T):
        """Overwrite initial velcovity by Boltzmann distribution.

        Args:
            T (float): Temperature in K
        """
        initial_velo = np.random.standard_normal(self.traj_x[0].shape)
        initial_velo = initial_velo * np.sqrt(self.const_kB * T / self.mass)
        initial_eng_kin = np.sum(np.sum(0.5 * self.mass * initial_velo * initial_velo, axis=-1), axis=-1, keepdims=True)
        initial_p = initial_velo * self.mass
        initial_T = 2 / 3 * initial_eng_kin / self.number_particles / self.const_kB

        self.traj_v[0] = initial_velo
        self.traj_E_kin[0] = initial_eng_kin
        self.traj_T[0] = initial_T
        self.traj_p[0] = initial_p
