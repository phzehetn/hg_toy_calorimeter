//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: B4aActionInitialization.cc 68058 2013-03-13 14:47:43Z gcosmo $
//
/// \file B4aActionInitialization.cc
/// \brief Implementation of the B4aActionInitialization class

#include "B4aActionInitialization.hh"
#include "B4PrimaryGeneratorAction.hh"
#include "B4RunAction.hh"
#include "B4aEventAction.hh"
#include "B4aSteppingAction.hh"
#include "B4DetectorConstruction.hh"
#include "PrimariesGenerator.hh"
#include "defines.h"
#include "B4aEventAction.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aActionInitialization::B4aActionInitialization
        (B4DetectorConstructionBase *detConstruction,
         B4aEventAction *event_action,
         B4PartGeneratorBase *primaries_generator)
        : G4VUserActionInitialization(),
          fDetConstruction(detConstruction),
          eventAction(event_action),
          primaries_generator(primaries_generator) {
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aActionInitialization::~B4aActionInitialization() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4aActionInitialization::BuildForMaster() const {
    B4aEventAction *ev;
    if (eventAction == nullptr)
        ev = new B4aEventAction(std::string(), false);
    else
        ev = eventAction;

    SetUserAction(new B4RunAction(primaries_generator, ev));
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4aActionInitialization::Build() const {
    SetUserAction(primaries_generator);

    B4aEventAction *ev;
    if (eventAction == nullptr)
        ev = new B4aEventAction(std::string(), false);
    else
        ev = eventAction;

    ev->setGenerator(primaries_generator);
    ev->setDetector(fDetConstruction);
    auto runact = new B4RunAction(primaries_generator, ev);
    SetUserAction(runact);
    SetUserAction(eventAction);
    auto steppingAction = new B4aSteppingAction(fDetConstruction, ev);
    SetUserAction(steppingAction);
    G4cout << "actions initialised" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
