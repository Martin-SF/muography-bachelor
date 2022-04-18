#include "EcoMug.h"

int main()
{
    EcoMug gen; // initialization of the class
    gen.SetUseSky(); // plane surface generation
    gen.SetSkySize({{10., 10.}}); // x and y size of the plane
    // (x,y,z) position of the center of the plane
    gen.SetSkyCenterPosition({{0., 0., 20.}});

    // The array storing muon generation position
    std::array<double, 3> muon_position;

    for (auto event = 0; event < number_of_events; ++event) {
        gen.Generate();
        muon_position = gen.GetGenerationPosition();
        double muon_p = gen.GetGenerationMomentum();
        double muon_theta = gen.GetGenerationTheta();
        double muon_phi = gen.GetGenerationPhi();
        double muon_charge = gen.GetCharge();
    }
}