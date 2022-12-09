from calo.calo_generator import CaloV3Generator


generator = CaloV3Generator()

specs = generator.generate()

nsen= 0
for layer_upper in specs['layers']:
    for layer in layer_upper:
        for eta_segment in layer['eta_segments']:
            if eta_segment['is_sensor']==True:
                nsen += int(eta_segment['phi_segments'])

print(nsen)
0/0