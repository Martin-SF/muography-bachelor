from build import EcoMug
from tqdm import tqdm
# import time
# start_time = time.time()


gen = EcoMug.EcoMug_Class()
# gen.SetUseSky()  # plane surface generation
# gen.SetUseCylinder()  # plane surface generation
gen.SetUseHSphere()  # plane surface generation
gen.SetSkySize((10., 10.))  # x and y size of the plane
gen.SetSkyCenterPosition(((0., 0., 20.)))  # (x,y,z) position of the center of the plane


number_of_events = int(1e6)
# for (int event = 0, event < number_of_events, ++event) {
for event in tqdm(range(int(number_of_events))):
    gen.Generate()
    gen.SetSeed(1234)
    # # muon_position = gen.GetGenerationPosition()
    # # muon_position.append(gen.GetGenerationPosition())
    # muon_position = gen.GetGenerationPosition()
    # muon_p = gen.GetGenerationMomentum()
    # muon_theta = gen.GetGenerationTheta()
    # muon_phi = gen.GetGenerationPhi()
    # muon_charge = gen.GetCharge()

# print("--- %s seconds ---" % (time.time() - start_time))
