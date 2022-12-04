//
// Created by Shah Rukh Qasim on 1/6/22.
//

#ifndef B4A_PRIMARIESGENERATOR_HH
#define B4A_PRIMARIESGENERATOR_HH

#include "B4PrimaryGeneratorAction.hh"
#include "G4PrimaryVertex.hh"
#include "B4PartGeneratorBase.hh"
#include "globals.hh"
#include <vector>
#include "Pythia8/Pythia.h"

#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"



class PrimariesGenerator : public B4PartGeneratorBase {
public:
    PrimariesGenerator(std::string pythiadata, int seed_0, int seed_1, int seed_2, int seed_3);
    void generate();
    void GenerateSingleVertex(G4PrimaryVertex* vertex);

    bool isJetGenerator() override;

    std::vector<G4String> generateAvailableParticles() const override;

    particles getParticle() const override;

    int isParticle(int i) const override;

    void GenerateWithPythia(G4Event *anEvent);
    void GenerateParticle(G4Event *anEvent);
    void GeneratePrimaries(G4Event *anEvent) override;

    G4PrimaryVertex * GeneratePUParticles();

    void SetNextToPU();
    void SetNextTo_qqbar_to_ttbar();
    void
    SetNextToParticle(double position_x, double position_y, double position_z, double direction_x, double direction_y,
                      double direction_z, int pdgid, double energy);

    enum class SimType { pu, singlepart, qqbar2ttbar };
    std::string getSimTypeAsString();
private:
    void init_pythia(Pythia8::Pythia*pythia, bool qqbar2ttbar);

protected:
    Pythia8::Pythia pythia_;
    Pythia8::Pythia pythia_qqbar2ttbar;
    fastjet::JetDefinition* jetDef_;
    Pythia8::RndmEngine* random_engine;

    double xorig_ = 0;
    double yorig_ = 0;
    double eta_min_lim = 1;
    double eta_max_lim = 4;

//    bool generatePU=true;


    SimType simulation_type;


    double particle_position_x = 0;
    double particle_position_y = 0;
    double particle_position_z = 0;

    double particle_direction_x = 0;
    double particle_direction_y = 0;
    double particle_direction_z = 0;
    double particle_energy = 0;
    int particle_pdgid = 0;

    G4ParticleGun* fParticleGun;


};



#endif //B4A_PRIMARIESGENERATOR_HH
