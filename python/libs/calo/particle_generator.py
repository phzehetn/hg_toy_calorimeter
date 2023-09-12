import numpy as np
import json

class ParticleGenerator():
    def pdgid_to_str(self, pdgid):
        data = {
            11: 'positron',
            -11: 'electron',
            22: 'gamma',
            211: 'pion_charged',
            111: 'pion_neutral',
            13: 'muon',
            15: 'tauon',
            2112: 'neutron',
            130: 'Klong',
        }

        return data[pdgid]

    def __init__(self, detector, range_energy=[1,200], particle_pdgid=[11, 22, 211, 111], range_eta=[3.1, 1.4]):
        self.particles = []
        self.range_x = [-0.001,0.001]
        self.range_y = [-0.001,0.001]
        self.range_z = [-0.1,+0.1]

        # self.particles_range = ['electron', 'muon', 'pion_charged', 'pion_neutral', 'gamma', 'positron']
        # self.particles_range = ['electron', 'muon', 'pion_charged']
        self.particle_pdgid = particle_pdgid
        self.particles_range = [self.pdgid_to_str(x) for x in self.particle_pdgid]
        # self.particles_range = ['muon']


        self.calo_end_z = detector['calo_end_z']
        self.calo_start_z = detector['calo_start_z']

        self.range_eta = [float(detector['calo_start_eta']), float(detector['calo_end_eta'])]
        # self.range_eta[0] += 0.001 # Give a buffer don't shoot at eta boundaries
        # self.range_eta[1] -= 0.003 # Give a buffer don't shoot at eta boundaries

        # self.range_eta = [3.1, 1.4]
        self.range_eta = range_eta


        self.range_phi = [0, 2*np.pi]
        # self.range_phi = [np.pi, 2 * np.pi]
        self.range_energy = range_energy



    def write_to_file(self, file):
        with open(file, 'w') as f:
            json.dump(self.particles, f)


    def eta_phi_z_to_x_y_z(self, eta, phi, z):
        theta = 2 * np.arctan(np.exp(-eta))
        s = z * np.tan(theta)

        x = np.sqrt(s ** 2 / (np.tan(phi) ** 2 + 1))
        x = -x if phi > np.pi else x
        y = x * np.tan(phi)

        return x, y, z


    # def x_y_z_to_eta_phi_z(self, x, y, z, return_theta=False):
    #     phi = np.arctan(y/x)
    #     s = np.sqrt(x**2 + y**2)
    #     theta = np.arctan(s/z)
    #     eta = - np.log(np.tan(theta/2))
    #
    #     if return_theta:
    #         return eta, phi, z, theta
    #     return eta, phi, z


    def generate_direction(self):
        direction_eta = np.random.uniform(self.range_eta[0], self.range_eta[1])
        direction_phi = np.random.uniform(self.range_phi[0], self.range_phi[1])

        direction_z = self.calo_end_z

        x, y, z = self.eta_phi_z_to_x_y_z(direction_eta, direction_phi, direction_z)

        vector = np.array([x,y,z])
        vector = vector / np.sqrt(np.sum(vector**2))

        vector = vector / 1000

        return vector.tolist(), direction_eta, direction_phi

    def generate_position(self, direction):

        direction = np.array(direction)
        position_x = 0#np.random.uniform(self.range_x[0], self.range_x[1])
        position_y = 0#np.random.uniform(self.range_y[0], self.range_y[1])
        position_z = 0#np.random.uniform(self.range_z[0], self.range_z[1])

        position = [position_x, position_y, position_z]
        position = np.array(position)

        translation_vector = ((self.calo_start_z - 0.001) / direction[2]) * direction

        # print(translation_vector)


        # print(self.calo_end_z, direction[2])

        position_translated = translation_vector + position

        return position_translated.tolist()

    def generate(self, from_iteraction_point=False):
        energy = np.random.uniform(self.range_energy[0], self.range_energy[1])

        direction, direction_eta, direction_phi = self.generate_direction()

        # Translates it to close to the calo
        position = [0.,0.,0.] if from_iteraction_point else self.generate_position(direction)

        x = np.random.randint(0, len(self.particles_range))
        id = self.particles_range[x]

        particle = dict()

        particle['position'] = position
        particle['direction'] = direction
        particle['direction_eta'] = direction_eta
        particle['direction_phi'] = direction_phi
        particle['energy'] = energy
        particle['id'] = id
        particle['pdgid'] = self.particle_pdgid[x]

        return particle

    def generate_randomly(self, num):
        for i in range(num):
            particle = self.generate()
            self.particles.append(particle)


