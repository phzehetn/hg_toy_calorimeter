//
// Created by Shah Rukh Qasim on 1/6/22.
//

#include <iostream>
#include "PrimariesGenerator.hh"

#include "FTFP_BERT.hh"
#include "QGSP_FTFP_BERT.hh"

#include "G4PhysicsListHelper.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4UserSpecialCuts.hh"
#include "G4StepLimiter.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4RunManager.hh"

#include "Pythia8/Pythia.h"

void enable_physlimits(void)
{
    // cf. Geant 4 HyperNews, Forum "Physics List", Message 129
    // http://geant4-hn.slac.stanford.edu:5090/HyperNews/public/get/phys-list/129.html

    G4UserSpecialCuts *specialCuts = new G4UserSpecialCuts;
    G4StepLimiter     *stepLimiter = new G4StepLimiter;

    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleTable::G4PTblDicIterator *particleIterator = particleTable->GetIterator();
    // make sure you have called "G4RunManager::Initialize()" before

    particleIterator->reset();
    while ((*particleIterator)()) {
        // iterate through all known particles

        G4ParticleDefinition *particleDefinition = particleIterator->value();
        G4ProcessManager *processManager = particleDefinition->GetProcessManager();

        if (processManager && !particleDefinition->IsShortLived() && particleDefinition->GetPDGCharge() != 0) {
            // the process manager should exist, but we don't need to limit short-lived particles or neutrals

            processManager->AddDiscreteProcess(stepLimiter);
            processManager->AddDiscreteProcess(specialCuts);
            // these transportation-related processes are always applicable

        }
    }
}

int main(int argc,char** argv) {
    auto runManager = new G4RunManager;

    std::cout<<"Hello, world!"<<std::endl;

    auto physicsList = new FTFP_BERT;
    physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    runManager->SetUserInitialization(physicsList);
//    enable_physlimits();


    auto gen = new PrimariesGenerator();
    gen->generate();

    delete runManager;
}

using namespace Pythia8;

//int main() {
//// Generator. Process selection. Tevatron initialization. Histogram.
//    Pythia pythia;
////    pythia.readString("Beams:idB = -2212");
////    pythia.readString("Beams:eCM = 1960.");
////    pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
////    pythia.readString("PhaseSpace:mHatMin = 80.");
////    pythia.readString("PhaseSpace:mHatMax = 120.");
////    pythia.init();
//
//
//
////    pythia.readString("Beams:idB = -2212");
////    pythia.readString("Beams:eCM = 1960.");
////    pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
////    pythia.readString("PhaseSpace:mHatMin = 80.");
////    pythia.readString("PhaseSpace:mHatMax = 120.");
////    pythia.init();
//
//    pythia.readString("Beams:eCM = 14000.");
//    pythia.readString("Init:showChangedParticleData = off");
//    pythia.readString("Init:showChangedSettings = on");
//    pythia.readString("Next:numberShowLHA = 0");
//    pythia.readString("Next:numberShowInfo = 0");
//    pythia.readString("Next:numberShowProcess = 0");
//    pythia.readString("Next:numberShowEvent = 0");
//
//    pythia.readString("Tune:pp 14");
//    pythia.readString("Tune:ee 7");
//    pythia.readString("MultipartonInteractions:ecmPow=0.03344");
//    pythia.readString("MultipartonInteractions:bProfile=2");
//    pythia.readString("MultipartonInteractions:pT0Ref=1.41");
//    pythia.readString("MultipartonInteractions:coreRadius=0.7634");
//    pythia.readString("MultipartonInteractions:coreFraction=0.63");
//    pythia.readString("ColourReconnection:range=5.176");
//    pythia.readString("SigmaTotal:zeroAXB=off");
//    pythia.readString("SpaceShower:alphaSorder=2");
//    pythia.readString("SpaceShower:alphaSvalue=0.118");
//    pythia.readString("SigmaProcess:alphaSvalue=0.118");
//    pythia.readString("SigmaProcess:alphaSorder=2");
//    pythia.readString("MultipartonInteractions:alphaSvalue=0.118");
//    pythia.readString("MultipartonInteractions:alphaSorder=2");
//    pythia.readString("TimeShower:alphaSorder=2");
//    pythia.readString("TimeShower:alphaSvalue=0.118");
//    pythia.readString("SigmaTotal:mode = 0");
//    pythia.readString("SigmaTotal:sigmaEl = 21.89");
//    pythia.readString("SigmaTotal:sigmaTot = 100.309");
//    //pythia.readString("PDF:pSet=LHAPDF6:NNPDF31_nnlo_as_0118");
//
//    //  pythia.readString("SoftQCD:nonDiffractive = on");
//    //  pythia.readString("SoftQCD:singleDiffractive = on");
//    //  pythia.readString("SoftQCD:doubleDiffractive = on");
//    pythia.readString("SoftQCD:inelastic = on");
//
//    pythia.readString("Tune:preferLHAPDF = 2");
//    pythia.readString("Main:timesAllowErrors = 10000");
//    pythia.readString("Check:epTolErr = 0.01");
//    pythia.readString("Beams:setProductionScalesFromLHEF = off");
////    pythia.readString("SLHA:keepSM = on");
//    pythia.readString("SLHA:minMassSM = 1000.");
//    pythia.readString("ParticleDecays:limitTau0 = on");
//    pythia.readString("ParticleDecays:tau0Max = 10");
//    pythia.readString("ParticleDecays:allowPhotonRadiation = on");
//
//    pythia.init();
//
//
//    Hist pTZ("dN/dpTZ", 100, 0., 100.);
//// Begin event loop. Generate event. Skip if error. List first one.
//    for (int iEvent = 0; iEvent < 1000; ++iEvent) {
//        if (!pythia.next()) continue;
//// Loop over particles in event. Find last Z0 copy. Fill its pT.
//        int iZ = 0;
//        for (int i = 0; i < pythia.event.size(); ++i)
//            if (pythia.event[i].id() == 23) iZ = i;
//        pTZ.fill( pythia.event[iZ].pT() );
//// End of event loop. Statistics. Histogram. Done.
//    }
//    pythia.stat();
//    cout << pTZ;
//    return 0;
//}