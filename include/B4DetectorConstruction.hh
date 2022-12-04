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
// $Id: B4DetectorConstruction.hh 75215 2013-10-29 16:07:06Z gcosmo $
// 
/// \file B4DetectorConstruction.hh
/// \brief Definition of the B4DetectorConstruction class

#ifndef B4DetectorConstruction_h
#define B4DetectorConstruction_h 1

#include <unordered_map>
#include "defines.h"

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

#include "G4ThreeVector.hh"

#include "SensorContainer.h"
#include "json/json.h"
#include "G4Cons.hh"

class G4VPhysicalVolume;
class G4GlobalMagFieldMessenger;
class G4Material;

/// Detector construction class to define materials and geometry.
/// The calorimeter is a box made of a given number of layers. A layer consists
/// of an absorber plate and of a detection gap. The layer is replicated.
///
/// Four parameters define the geometry of the calorimeter :
///
/// - the thickness of an absorber plate,
/// - the thickness of a gap,
/// - the number of layers,
/// - the transverse size of the calorimeter (the input face is a square).
///
/// In addition a transverse uniform magnetic field is defined 
/// via G4GlobalMagFieldMessenger class.

G4double computeTheta(const G4double& eta );

double etaToR(const G4double& eta, const G4double& z);
G4Cons * createCons(
        G4String name,
        G4double start_eta,
        G4double eta_width,
        G4double start_z,
        G4double z_length,
        G4double start_phi,
        G4double end_phi);


class B4DetectorConstructionBase : public G4VUserDetectorConstruction {
public:
    virtual const std::vector<std::shared_ptr<SensorContainer>>* getActiveSensors()const = 0;
	virtual const std::unordered_map<int, std::unordered_map<int, std::shared_ptr<SensorContainer>>> *
	getIndexedSensorContainers() const = 0;
    virtual double getCaloStartZ()const = 0;
    virtual G4VPhysicalVolume* getWorld() const = 0;
};

class B4DetectorConstruction : public B4DetectorConstructionBase
{
protected:
	std::unordered_map<int, std::unordered_map<int, std::shared_ptr<SensorContainer>>> indexedSensorContainers;
public:
	virtual const std::unordered_map<int, std::unordered_map<int, std::shared_ptr<SensorContainer>>> *
	getIndexedSensorContainers() const;

public:
    B4DetectorConstruction(Json::Value& value);
    virtual ~B4DetectorConstruction();

  public:
    enum geometry{
    	standard,
		homogenous,
		homogenous_ecal_only,
		ecal_only,
		ecal_only_hi_granular,
		hcal_only_irregular,
		ecal_only_irregular,
		homogenous_no_tracker,
		fullendcap
    };

    G4VPhysicalVolume *getWorld() const override;

    double getCaloStartZ() const override;

    virtual G4VPhysicalVolume* Construct();
    virtual void ConstructSDandField();

    void  DefineGeometry(geometry g);

    bool isActiveVolume(G4VPhysicalVolume*)const;

    virtual const std::vector<std::shared_ptr<SensorContainer>>* getActiveSensors()const;

     
  private:
    // methods
    //
    void DefineMaterials();
    G4VPhysicalVolume* DefineVolumes();

    void createActiveCellWheel(G4Material *material, G4double start_eta, G4double end_eta,
                               G4double start_z, G4double end_z, G4double nphi, G4int layernum,
                               G4int cellnum, G4LogicalVolume *layerLV, G4ThreeVector position,
                               G4double z_step_max, G4double pre_absorber_thickness,
                               int active_layer_num);


	void createAbsorberCellWheel(G4Material *material, G4double start_eta, G4double end_eta,
									 G4double start_z, G4double end_z, G4double nphi, G4int layernum,
									 G4int cellnum, G4LogicalVolume *layerLV, G4ThreeVector position,
									 G4double z_step_max);

    G4Material* getMaterial(std::string name);


	void createLayer(Json::Value &layer, int layernum);
	void createAbsorber(Json::Value &layer, int layernum);
	void createActiveLayer(Json::Value &layer, int layernum);
  
    void createCalo(G4LogicalVolume *worldLV, G4String name);
    // data members
    //
    static G4ThreadLocal G4GlobalMagFieldMessenger*  fMagFieldMessenger; 
                                      // magnetic field messenger

    std::vector<std::shared_ptr<SensorContainer>> activecells_;

    G4bool  fCheckOverlaps; // option to activate checking of volumes overlaps

    G4double layerThicknessEE,layerThicknessHB;
    G4double Calo_start_eta=-1;
    G4double Calo_end_eta=-1;
    G4double m_calo_start_z=-1;
    std::vector<G4double> layerThicknesses,layerAbsorberFractions;
    G4int Ncalowedges;
    std::vector<G4int> layerGranularity;
    std::vector<G4int> layerSplitGranularity;
    G4double calorSizeXY;
    G4Material * m_vacuum, *m_pb, *m_pbtungsten, *m_silicon, *m_cu, *m_stainlesssteel, *m_air;
    G4int nofEELayers,nofHB, noTrackLayers;
    G4double calorThickness;

    Json::Value& m_detector_specs;


    G4double limit_in_calo_time_max_, limit_in_calo_energy_max_;
    G4double limit_world_time_max_,limit_world_energy_max_;

	G4LogicalVolume * m_worldLV;

    G4VPhysicalVolume* m_worldPV;


};

// inline functions

inline const std::vector<std::shared_ptr<SensorContainer>>* B4DetectorConstruction::getActiveSensors()const{
	return &activecells_;
}

     

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

