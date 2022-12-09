import numpy as np

MATERIAL_COPPER = 'G4_Cu'
# MATERIAL_COPPER_TUNGSTON = 'G4_CuW'
MATERIAL_SILICON = 'G4_Si'
# MATERIAL_PBW = 'G4_PbWO4'
MATERIAL_PB = 'G4_Pb'
MATERIAL_VACUUM = 'Galactic'
MATERIAL_AIR = 'Air'
MATERIAL_STAINLESS_STEEL = 'StainlessSteel'


class CaloGenerator():
    def __init__(self):
        self.num_ee_layers = 14
        self.num_hb_layers_1 = 12
        self.num_hb_layers_2 = 16
        self.total_layers = self.num_ee_layers * 2 + self.num_hb_layers_1 + self.num_hb_layers_2

        self.detector = dict()
        self.detector['world_size_xy'] = 4
        self.detector['world_size_z'] = 16
        self.detector['material'] = MATERIAL_VACUUM
        self.detector['layers'] = []

        self.start_eta = 1.5
        self.end_eta = 3.0


    def generate(self):
        num_active_layer = 0

        start_delta_eta_phi = 0.03
        end_delta_eta_phi = 0.08

        delta = lambda num_active_layer: (end_delta_eta_phi - start_delta_eta_phi) * num_active_layer / (self.total_layers-1) + start_delta_eta_phi

        print(delta(0), delta(55))
        c = 2#, TODO: 4

        l_num_eta_cuts = lambda num_active_layer: int(np.ceil(
            np.sqrt((c ** 2 * (self.end_eta - self.start_eta) ** 2 + 4 * np.pi ** 2) / (delta(num_active_layer) ** 2 * c ** 2))))
        l_num_phi_cuts = lambda num_active_layer: l_num_eta_cuts(num_active_layer) * c


        #
        # exit()

        """
        Thickness: 200 micrometer
        """

        now_z = 3.2
        self.detector['calo_start_z'] = now_z
        num_sensors = 0
        last_active_z = now_z
        for i in range(self.num_ee_layers):
            print("X", now_z)
            layers = []

            thickness = 0.006
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_ee_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z, active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap


            # thickness = 0.00014
            # gap = 0.00001
            # layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER_TUNGSTON))
            # now_z += thickness
            # now_z += gap

            thickness = 0.006
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            # thickness = 0.00014
            # gap = 0.00001
            # layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER_TUNGSTON))
            # now_z += thickness
            # now_z += gap

            thickness = 0.006
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_ee_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z, active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1


            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap


            thickness = 0.0021
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_PB, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0006
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap


            thickness = 0.0001
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap


            self.detector['layers'].append(layers)

        self.detector['ecal_end_z'] = now_z
        print("ECal finished", now_z)

        for i in range(self.num_hb_layers_1):
            print("Y", now_z)
            layers = []
            thickness = 0.035
            gap = 0.0001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0032
            gap = 0.0001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_hb_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z, active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            num_active_layer += 1
            now_z += gap


            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.006
            gap = 0.0001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        for i in range(self.num_hb_layers_2):
            print("Z", now_z)
            layers = []
            thickness = 0.068
            gap = 0.0001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0032
            gap = 0.0001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            print("DELTA",num_active_layer,delta(num_active_layer))
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_hb_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z, active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.006
            gap = 0.0001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        print(now_z)
        print("Num sensors", num_sensors)
        print("Num active layers", num_active_layer, )

        self.detector['calo_end_z'] = now_z
        self.detector['calo_start_eta'] = self.start_eta
        self.detector['calo_end_eta'] = self.end_eta

        return self.detector


    def create_segments(self, num_segments, num_phi=40, is_sensor=False):
        segments = []

        x = np.linspace(self.start_eta, self.end_eta, num_segments+1)

        for i in range(len(x)-1):
            this_start_eta = x[i]
            this_end_eta = x[i+1]
            segments.append({'start_eta':this_start_eta, 'end_eta':this_end_eta,'phi_segments':num_phi, 'is_sensor':is_sensor})


        return segments


    def create_absorber(self, start_z, end_z, material=MATERIAL_COPPER, n_z_divides=10, num_eta_cuts=7):
        specs = dict()
        specs['material'] = material
        specs['eta_segments'] = self.create_segments(num_eta_cuts)
        specs['start_z'] = start_z
        specs['end_z'] = end_z
        specs['offset'] = 0
        specs['type'] = 'absorber'
        specs['z_step_max'] = (end_z-start_z) / float(n_z_divides)
        return specs


    def create_ee_layer(self, start_z, end_z, material=MATERIAL_SILICON, num_phi_cuts=40, num_eta_cuts=24, n_z_divides=10, last_active_z=-1.,active_layer_num=-1):
        specs = dict()
        specs['material'] = material
        specs['eta_segments'] = self.create_segments(num_eta_cuts, is_sensor=True, num_phi=num_phi_cuts)
        specs['start_z'] = start_z
        specs['end_z'] = end_z
        specs['offset'] = 0
        specs['type'] = 'ee'
        specs['z_step_max'] = (end_z-start_z) / float(n_z_divides)
        specs['pre_absorber_thickness'] = start_z - last_active_z
        specs['active_layer_num'] = active_layer_num

        return specs

    def create_hb_layer(self, start_z, end_z, material=MATERIAL_SILICON, num_phi_cuts=40, num_eta_cuts=24, n_z_divides=10, last_active_z=-1., active_layer_num=-1):
        specs = dict()
        specs['material'] = material
        specs['eta_segments'] = self.create_segments(num_eta_cuts, is_sensor=True, num_phi=num_phi_cuts)
        specs['start_z'] = start_z
        specs['end_z'] = end_z
        specs['offset'] = 0
        specs['type'] = 'hb'
        specs['z_step_max'] = (end_z-start_z) /  float(n_z_divides)
        specs['pre_absorber_thickness'] = start_z - last_active_z
        specs['active_layer_num'] = active_layer_num


        return specs



class CaloV2Generator(CaloGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        num_active_layer = 0

        start_delta_eta_phi = 0.02
        end_delta_eta_phi = 0.07

        delta = lambda num_active_layer: (end_delta_eta_phi - start_delta_eta_phi) * num_active_layer / (
                    self.total_layers - 1) + start_delta_eta_phi
        c = 4
        l_num_eta_cuts = lambda num_active_layer: int(np.ceil(
            np.sqrt((c ** 2 * (self.end_eta - self.start_eta) ** 2 + 4 * np.pi ** 2) / (
                        delta(num_active_layer) ** 2 * c ** 2))))
        l_num_phi_cuts = lambda num_active_layer: l_num_eta_cuts(num_active_layer) * c


        """
        Thickness: 200 micrometer
        """

        now_z = 3.2
        self.detector['calo_start_z'] = now_z
        num_sensors = 0
        last_active_z = now_z
        for i in range(self.num_ee_layers):
            print("X", now_z)
            layers = []

            thickness = 0.0002
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_ee_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            # thickness = 0.00014
            # gap = 0.00001
            # layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER_TUNGSTON))
            # now_z += thickness
            # now_z += gap

            thickness = 0.006
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            # thickness = 0.00014
            # gap = 0.00001
            # layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER_TUNGSTON))
            # now_z += thickness
            # now_z += gap

            thickness = 0.0002
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_ee_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0021
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_PB, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0006
            gap = 0.00001
            layers.append(
                self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0001
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        self.detector['ecal_end_z'] = now_z
        print("ECal finished", now_z)

        for i in range(self.num_hb_layers_1):
            print("Y", now_z)
            layers = []
            thickness = 0.035
            gap = 0.0001
            layers.append(
                self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0002
            gap = 0.0001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_hb_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            num_active_layer += 1
            now_z += gap

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.006
            gap = 0.0001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        for i in range(self.num_hb_layers_2):
            print("Z", now_z)
            layers = []
            thickness = 0.068
            gap = 0.0001
            layers.append(
                self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0002
            gap = 0.0001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            print("DELTA", num_active_layer, delta(num_active_layer))
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_hb_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.006
            gap = 0.0001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        print(now_z)
        print("Num sensors", num_sensors)
        print("Num active layers", num_active_layer, )

        self.detector['calo_end_z'] = now_z
        self.detector['calo_start_eta'] = self.start_eta
        self.detector['calo_end_eta'] = self.end_eta

        return self.detector


class CaloV3Generator(CaloGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        num_active_layer = 0

        start_delta_eta_phi = 0.014
        end_delta_eta_phi = 0.022
        n_layer_constant_size = 40

        delta = lambda num_active_layer: (end_delta_eta_phi - start_delta_eta_phi) * min(num_active_layer, n_layer_constant_size-1) / (
                    n_layer_constant_size - 1) + start_delta_eta_phi

        c = 4
        l_num_eta_cuts = lambda num_active_layer: int(np.ceil(
            np.sqrt((c ** 2 * (self.end_eta - self.start_eta) ** 2 + 4 * np.pi ** 2) / (
                        delta(num_active_layer) ** 2 * c ** 2))))
        l_num_phi_cuts = lambda num_active_layer: l_num_eta_cuts(num_active_layer) * c


        """
        Thickness: 200 micrometer
        """
        now_z = 3.2
        self.detector['calo_start_z'] = now_z
        num_sensors = 0
        last_active_z = now_z
        for i in range(self.num_ee_layers):
            print("X", now_z)
            layers = []

            thickness = 0.0002
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_ee_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            # thickness = 0.00014
            # gap = 0.00001
            # layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER_TUNGSTON))
            # now_z += thickness
            # now_z += gap

            thickness = 0.006
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            # thickness = 0.00014
            # gap = 0.00001
            # layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER_TUNGSTON))
            # now_z += thickness
            # now_z += gap

            thickness = 0.0002
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_ee_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0021
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_PB, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0006
            gap = 0.00001
            layers.append(
                self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0001
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        self.detector['ecal_end_z'] = now_z
        print("ECal finished", now_z)

        for i in range(self.num_hb_layers_1):
            print("Y", now_z)
            layers = []
            thickness = 0.035
            gap = 0.00001
            layers.append(
                self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0002
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_hb_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            num_active_layer += 1
            now_z += gap

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.006
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        for i in range(self.num_hb_layers_2):
            print("Z", now_z)
            layers = []
            thickness = 0.068
            gap = 0.00001
            layers.append(
                self.create_absorber(now_z, now_z + thickness, material=MATERIAL_STAINLESS_STEEL, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.0002
            gap = 0.00001
            num_eta_cuts = l_num_eta_cuts(num_active_layer)
            num_phi_cuts = l_num_phi_cuts(num_active_layer)
            print("DELTA", num_active_layer, delta(num_active_layer))
            num_sensors += num_eta_cuts * num_phi_cuts
            layers.append(
                self.create_hb_layer(now_z, now_z + thickness, material=MATERIAL_SILICON, num_phi_cuts=num_phi_cuts,
                                     num_eta_cuts=num_eta_cuts, n_z_divides=5, last_active_z=last_active_z,
                                     active_layer_num=num_active_layer))
            now_z += thickness
            last_active_z = now_z
            now_z += gap
            num_active_layer += 1

            thickness = 0.0015
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_AIR, n_z_divides=5))
            now_z += thickness
            now_z += gap

            thickness = 0.006
            gap = 0.00001
            layers.append(self.create_absorber(now_z, now_z + thickness, material=MATERIAL_COPPER, n_z_divides=5))
            now_z += thickness
            now_z += gap

            self.detector['layers'].append(layers)

        print(now_z)
        print("Num sensors", num_sensors)
        print("Num active layers", num_active_layer, )

        self.detector['calo_end_z'] = now_z
        self.detector['calo_start_eta'] = self.start_eta
        self.detector['calo_end_eta'] = self.end_eta

        return self.detector