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
// $Id: B4DetectorConstruction.cc 101905 2016-12-07 11:34:39Z gunter $
// 
/// \file B4DetectorConstruction.cc
/// \brief Implementation of the B4DetectorConstruction class

#include "B4DetectorConstruction.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"

#include "G4RotationMatrix.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4GlobalMagFieldMessenger.hh"
#include "G4AutoDelete.hh"
#include "G4UserLimits.hh"
#include "G4PVDivision.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"

#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4PropagatorInField.hh"
#include "json/json.h"

#include "G4Cons.hh"
#include <cmath>

#include "SensorContainer.h"

#include <cstdlib>
//#include "Math/Vector3D.h"

//#define USEDIVISION

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ThreadLocal 
G4GlobalMagFieldMessenger* B4DetectorConstruction::fMagFieldMessenger = nullptr; 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

template<class T>
static G4String createString(const T& i){
	std::stringstream ss;
	ss << i;
	std::string number=ss.str();
	return number;
}


B4DetectorConstruction::B4DetectorConstruction(Json::Value& value)
: B4DetectorConstructionBase(),
  fCheckOverlaps(false),
  m_vacuum(0),
  m_pb(0),
  m_pbtungsten(0),
  m_silicon(0),
  m_cu(0),
  m_detector_specs(value)
{


    limit_in_calo_time_max_=500*ns;//this could be more low energy stuff
//    limit_in_calo_energy_max_=500*keV;
	limit_in_calo_energy_max_=0.1*MeV;
    limit_world_time_max_=500*ns; //either got there or not (30ns should be easily sufficient
    limit_world_energy_max_=100*eV;

}

G4VPhysicalVolume* B4DetectorConstruction::Construct()
{
	//DefineGeometry(homogenous_ecal_only);
	//DefineGeometry(hcal_only_irregular);
	DefineGeometry(fullendcap);
	// Define materials
	DefineMaterials();

	// Define volumes
	return DefineVolumes();
}

void  B4DetectorConstruction::DefineGeometry(geometry geo){

    layerAbsorberFractions.clear();
    layerThicknesses.clear();
    layerGranularity.clear();
	if(geo == fullendcap){
		calorThickness=2200*mm;

        nofEELayers = 14;
        Ncalowedges=96;
        nofHB=18;
        int etasegments=24;

        Calo_start_eta=1.5;
        Calo_end_eta=3.0;
//        m_calo_start_z=320*cm;

        for(int i=0;i<nofEELayers+nofHB;i++){
            if(i<nofEELayers){
                layerThicknesses.push_back(10.382*mm + 0.300*mm);////absorber plus silicon, this makes 1.85 X0 per layer
                layerAbsorberFractions.push_back(0.9719153717);
            }
            else{//copper absorber: 15.32 cm == 1 l_0 , ECal part already has about 1. Add 9 more// 1/2 per layer
                layerThicknesses.push_back(15.32*cm / 2. + 0.300*mm);
                layerAbsorberFractions.push_back(0.9960988296);
            }

		    layerGranularity.push_back(etasegments);
		}

		layerThicknessEE=26*cm / (float)nofEELayers;
		layerThicknessHB=(calorThickness-nofEELayers*layerThicknessEE)/(float)nofHB; //100*mm;

	}
	else{
	    G4ExceptionDescription msg;
	            msg << "Geometry not supported in this branch";
	            G4Exception("B4DetectorConstruction::DefineGeometry()",
	                    "MyCode0001", FatalException, msg);
	}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4DetectorConstruction::~B4DetectorConstruction()
{ 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


G4double computeTheta(const G4double& eta )
        { return 2. * atan( exp( -eta ) ); }

double etaToR(const G4double& eta, const G4double& z){
    return 2 * z * exp(-eta);
}
G4Cons * createCons(
        G4String name,
        G4double start_eta,
        G4double eta_width,
        G4double start_z,
        G4double z_length,
        G4double start_phi,
        G4double end_phi){



  //  G4cout << "created Cons "<< name <<"with size (eta, deta, phi,endphi,z,dz): "
  //          << start_eta <<", "<< eta_width <<", "
  //          << start_phi << ", "<<end_phi<<", "
  //          << start_z << ", "<< z_length  << G4endl;
  //
  //  G4cout << "(pRmin1,pRmax1) " << etaToR(start_eta+eta_width ,start_z) <<", " << etaToR(start_eta ,start_z)<< G4endl;
  //  G4cout << "(pRmin2,pRmax2) " << etaToR(start_eta+eta_width ,start_z+z_length) <<", " << etaToR(start_eta ,start_z+z_length)<< G4endl;


    return new G4Cons(
            name,
            etaToR(start_eta+eta_width ,start_z),
            etaToR(start_eta ,start_z),
            etaToR(start_eta+eta_width ,start_z+z_length),
            etaToR(start_eta ,start_z+z_length),
            z_length/2.,
            start_phi,
            end_phi);
}


void B4DetectorConstruction::createAbsorberCellWheel(G4Material *material, G4double start_eta, G4double end_eta,
													 G4double start_z, G4double end_z, G4double nphi, G4int layernum,
													 G4int cellnum, G4LogicalVolume *layerLV, G4ThreeVector position,
													 G4double z_step_max) {
	G4String name = createString(layernum)+"_"+createString(cellnum);
	double eta_width = end_eta - start_eta;
	double z_length = end_z - start_z;
	auto absWheelS = createCons("Abswheel_"+name,
								start_eta,
								eta_width,
								start_z,
								z_length,
								0,
								2.*M_PI);

	auto  abswheelLV = new G4LogicalVolume(
			absWheelS,           // its solid
			material,  // its material
			"Abswheel_LV_"+name);         // its name

	auto abswheelPV = new G4PVPlacement(
			0,                // no rotation
			position+ G4ThreeVector(0,0,z_length/2.), // its position
			abswheelLV,       // its logical volume
			"Aabswheel_PV_"+name,           // its name
			layerLV,          // its mother  volume
			false,            // no boolean operation
			0,                // copy number
			fCheckOverlaps);  // checking overlaps


	abswheelLV->SetUserLimits(new G4UserLimits(z_step_max, DBL_MAX,
											   limit_in_calo_time_max_,limit_in_calo_energy_max_));
}


void B4DetectorConstruction::createActiveCellWheel(G4Material *material, G4double start_eta, G4double end_eta,
                                                   G4double start_z, G4double end_z, G4double nphi, G4int layernum,
                                                   G4int cellnum, G4LogicalVolume *layerLV, G4ThreeVector position,
                                                   G4double z_step_max, G4double pre_absorber_thickness,
                                                   int active_layer_num) {
	G4double active_z_length=end_z - start_z;
	G4double midz = active_z_length/2.;
	G4double thickness = end_z - start_z;


	//create the volume

	G4String name = createString(layernum)+"_"+createString(cellnum);

	G4LogicalVolume* activewheelLV=layerLV;
	G4PVPlacement * activewheelPV=0;
	auto activewheelS = createCons("Activewheel_"+name,
								   start_eta,
								   end_eta - start_eta,
								   start_z,
								   active_z_length,
								   0,
								   2.*M_PI);
    double surfaceArea = activewheelS->GetSurfaceArea();

	activewheelLV = new G4LogicalVolume(
			activewheelS,           // its solid
			material,  // its material
			"Activewheel_LV_"+name);         // its name


	activewheelPV = new G4PVPlacement(
			0,                // no rotation
			position+ G4ThreeVector(0,0,thickness/2.), // its position
			activewheelLV,       // its logical volume
			"Activewheel_PV_"+name,           // its name
			layerLV,          // its mother  volume
			false,            // no boolean operation
			0,                // copy number
			fCheckOverlaps);  // checking overlaps

	G4cout << "placed active wheel at z_start = " << start_z << " ends " <<  end_z << G4endl;


	auto cellS   = createCons("Cell_"+name,
							  start_eta,
							  (end_eta - start_eta),
							  start_z,
							  active_z_length,
							  0,
							  2.*M_PI/(double)nphi);

	auto cellLV  = new G4LogicalVolume(
			cellS,           // its solid
			material,  // its material
			"Cell_LV_"+name);         // its name


	cellLV->SetUserLimits(new G4UserLimits(z_step_max,DBL_MAX,
										   limit_in_calo_time_max_,limit_in_calo_energy_max_));


#ifdef USEDIVISION
	G4VPhysicalVolume* activeMaterial
            = new G4PVDivision("Cell_rep_"+name, cellLV,layerLV, kPhi, nphi,0.);
#else
	G4VPhysicalVolume* activeMaterial
	= new G4PVReplica("Cell_rep_"+name, cellLV,
									 activewheelPV, kPhi, nphi, 2.*M_PI/(double)nphi,0.);
#endif


	//check overlaps thoroughly
	if (fCheckOverlaps)
		activeMaterial-> CheckOverlaps(1000000, 0., true);

	G4int maxcopies = activeMaterial->GetMultiplicity();

	//only consider active material here

	std::unordered_map<int, std::shared_ptr<SensorContainer>> thisVolSensorContainersDict;

//	std::cout<<"CHECK THIS: "<<start_eta<<" " <<end_eta<<" "<<start_z<<" "<<end_z<<std::endl;
	double width_rad = 2.*M_PI/(double)nphi*rad;
    double surfaceAreaSensor = surfaceArea / maxcopies;
	for(G4int i =0;i<maxcopies;i++){
		double phi= width_rad*(double)i; //offset doesn't really matter

		double x = cos(phi)*etaToR(start_eta+(end_eta - start_eta)/2. ,start_z+thickness/2.);
		double y = sin(phi)*etaToR(start_eta+(end_eta - start_eta)/2. ,start_z+thickness/2.);

        std::shared_ptr<SensorContainer> sc(new SensorContainer(
                activeMaterial,
                start_eta + (end_eta - start_eta) / 2.,//sensor size   // G4double dimxy
                phi,                              // G4double dimz,
                surfaceAreaSensor,       // G4double area,
                x,                   // G4double posx,
                y,                   // G4double posy,
                start_z + thickness / 2.,                   // G4double posz,
                pre_absorber_thickness,
                active_layer_num,
                i //copyno
                , thickness, width_rad, end_eta - start_eta));
		thisVolSensorContainersDict[i] = sc;

		activecells_.push_back(sc);
	}
	indexedSensorContainers[activeMaterial->GetInstanceID()] = thisVolSensorContainersDict;


}

G4Material* B4DetectorConstruction::getMaterial(std::string name) {
	if (name ==std::string("G4_Cu")) {
		return m_cu;
	}
	else if (name ==std::string("G4_Si")) {
		return m_silicon;
	}
	if (name ==std::string("G4_PbWO4")) {
		return m_pbtungsten;
	}
	if (name ==std::string("Galactic")) {
		return m_vacuum;
	}
	if (name ==std::string("Air")) {
		return m_air;

	}
	if (name ==std::string("StainlessSteel")) {
		return m_stainlesssteel;

	}
	if (name ==std::string("G4_Pb")) {
		return m_pb;

	}

	return nullptr;
}

void B4DetectorConstruction::createAbsorber(Json::Value &layer, int layernum) {
	G4Material* material = getMaterial(layer["material"].asString());
	int cellnum = 0;
	for (auto &eta_segment : layer["eta_segments"]) {
		createAbsorberCellWheel(material,
								eta_segment["start_eta"].asDouble(),
								eta_segment["end_eta"].asDouble(),
								layer["start_z"].asDouble() * m,
								layer["end_z"].asDouble() * m,
								eta_segment["phi_segments"].asDouble(),
								layernum,
								cellnum, m_worldLV, G4ThreeVector(0, 0, layer["start_z"].asDouble() * m),
								layer["z_step_max"].asDouble() * m);
		cellnum += 1;
	}
}

void B4DetectorConstruction::createActiveLayer(Json::Value &layer, int layernum) {
	G4Material* material = getMaterial(layer["material"].asString());
	int cellnum = 0;


	for (auto &eta_segment : layer["eta_segments"]) {

        createActiveCellWheel(material,
                              eta_segment["start_eta"].asDouble(),
                              eta_segment["end_eta"].asDouble(),
                              layer["start_z"].asDouble() * m,
                              layer["end_z"].asDouble() * m,
                              eta_segment["phi_segments"].asDouble(),
                              layernum,
                              cellnum, m_worldLV, G4ThreeVector(0, 0, layer["start_z"].asDouble() * m),
                              layer["z_step_max"].asDouble() * m, layer["pre_absorber_thickness"].asDouble() * m, layer["active_layer_num"].asInt());
		cellnum += 1;
	}
}

void B4DetectorConstruction::createLayer(Json::Value &layer, int layernum) {

	std::string layer_type = layer["type"].asString();
	std::cout<<"Z Values: "<<layer["start_z"].asDouble()<<" to "<< layer["end_z"].asDouble() <<std::endl;

    if (m_calo_start_z == -1) {
        m_calo_start_z = layer["start_z"].asDouble() * m;
    }

	if (std::string("absorber") == layer_type) {
		createAbsorber(layer, layernum);
	}
	else {
		createActiveLayer(layer, layernum);
	}
}

void B4DetectorConstruction::createCalo(G4LogicalVolume *worldLV, G4String name) {

	m_worldLV = worldLV;

	int layernum = 0;
	auto layers_upper = m_detector_specs["layers"];
	for (auto &layer_upper: layers_upper) {
		for (auto &layer: layer_upper) {

			createLayer(layer, layernum);
			layernum += 1;
		}
	}
}

void B4DetectorConstruction::DefineMaterials()
{ 
	// Lead material defined using NIST Manager
	auto nistManager = G4NistManager::Instance();
	nistManager->FindOrBuildMaterial("G4_Pb");
    nistManager->FindOrBuildMaterial("G4_Cu");
	nistManager->FindOrBuildMaterial("G4_PbWO4");
    nistManager->FindOrBuildMaterial("G4_Si");
   // nistManager->FindOrBuildMaterial("G4_Air");

	//nistManager->ListMaterials("all");

	// Liquid argon material
	G4double a;  // mass of a mole;
	G4double z;  // z=mean number of protons;
	G4double density;
	new G4Material("liquidArgon", z=18., a= 39.95*g/mole, density= 1.390*g/cm3);
	// The argon by NIST Manager is a gas with a different density

	// Vacuum
	new G4Material("Galactic", z=1., a=1.01*g/mole,density= universe_mean_density,
			kStateGas, 2.73*kelvin, 3.e-18*pascal);

	// Print materials
	//G4cout << *(G4Material::GetMaterialTable()) << G4endl;


	// Get materials
	m_vacuum = G4Material::GetMaterial("Galactic"); // "Galactic"
	m_pb = G4Material::GetMaterial("G4_Pb");
	m_pbtungsten = G4Material::GetMaterial("G4_PbWO4");
	m_silicon = G4Material::GetMaterial("G4_Si");
	m_cu = G4Material::GetMaterial("G4_Cu");
//	m_cuw = G4Material::GetMaterial("G4_CuW");


//	m_stainlesssteel = G4Material::GetMaterial("G4_STAINLESS-STEEL");

	G4Element* Mn  = new G4Element("Manganese", "Mn", z=25, a=    54.94*g/mole);
	G4Element* Cr  = new G4Element("Chromium",  "Cr", z=24, a=    52.00*g/mole);
	G4Element* Ni  = new G4Element("Nickel",    "Ni", z=28, a=    58.70*g/mole);

	a=16.*g/mole;
	auto elO=new G4Element("Oxygen","O2",8.,a);
	a=14.01*g/mole;
	auto elN=new G4Element("Nitrogen","N2",7.,a);

	density = 1.290*mg/cm3;


	m_air = new G4Material("Air",density, 2);
	m_air->AddElement(elN, 0.7);
	m_air->AddElement(elO, 0.3);

	m_stainlesssteel = new G4Material("StainlessSteel",8.02*g/cm3, 5);
	m_stainlesssteel->AddElement(Mn, 0.02);
	m_stainlesssteel->AddMaterial(nistManager->FindOrBuildMaterial("G4_Si"), 0.01);
	m_stainlesssteel->AddElement(Cr, 0.19);
	m_stainlesssteel->AddElement(Ni, 0.10);
	m_stainlesssteel->AddMaterial(nistManager->FindOrBuildMaterial("G4_Fe"), 0.68);


	std::cout<<"CHECK MATERIALS\n\n"<<std::endl;
//	std::cout<<m_cuw<<std::endl;
	std::cout<<"CHECK MATERIALS END\n\n"<<std::endl;

	if ( ! m_vacuum || ! m_pb || ! m_pbtungsten || !m_silicon ) {
		G4ExceptionDescription msg;
		msg << "Cannot retrieve materials already defined.";
		G4Exception("B4DetectorConstruction::DefineVolumes()",
				"MyCode0001", FatalException, msg);
	}

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* B4DetectorConstruction::DefineVolumes()
{
	// Geometry parameters
	//calorSizeXY  = 200*cm;






    //auto calorThickness = nofEELayers * layerThicknessEE + nofHB*layerThicknessHB;
    G4double worldSizeXY = 6. * m;
	G4double worldSizeZ  = 16. * m;



	//
	// World
	//
	auto worldS
	= new G4Box("World",           // its name
			worldSizeXY/2, worldSizeXY/2, worldSizeZ); // its size

	auto worldLV
	= new G4LogicalVolume(
			worldS,           // its solid
			m_vacuum,  // its material
			"World");         // its name

	worldLV->SetUserLimits(new G4UserLimits(
	        worldSizeZ/10., //max step length
            worldSizeZ*50., //max track length
            limit_world_time_max_, //max track time
            limit_world_energy_max_)); //min track energy


	auto worldPV
	= new G4PVPlacement(
			0,                // no rotation
			G4ThreeVector(),  // at (0,0,0)
			worldLV,          // its logical volume
			"World",          // its name
			0,                // its mother  volume
			false,            // no boolean operation
			0,                // copy number
			fCheckOverlaps);  // checking overlaps

	//
	// Calorimeter
	//
    createCalo(worldLV, "");



	G4cout << "created in total "<< activecells_.size()<<" sensors" <<G4endl;

	//
	// Visualization attributes
	//


	worldLV->SetVisAttributes (G4VisAttributes::GetInvisible());

	double g=0;
	for(auto& v: activecells_){
	    auto simpleBoxVisAtt= new G4VisAttributes(G4Colour(.5,g,.0));
	    simpleBoxVisAtt->SetVisibility(true);
	    simpleBoxVisAtt->SetForceSolid(true);
		v->getVol()->GetLogicalVolume()->SetVisAttributes(simpleBoxVisAtt);
		g+= 1./(double)activecells_.size();
	}
	//
	// Always return the physical World
	//

    m_worldPV = worldPV;
	return worldPV;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4DetectorConstruction::ConstructSDandField()
{ 
	// Create global magnetic field messenger.
	// Uniform magnetic field is then created automatically if
	// the field value is not zero.
	G4ThreeVector fieldValue(0,0,0);// no need for field here 1.*tesla);
	fMagFieldMessenger = new G4GlobalMagFieldMessenger(fieldValue);
	fMagFieldMessenger->SetVerboseLevel(2);

	// Register the field messenger for deleting
	G4AutoDelete::Register(fMagFieldMessenger);


	auto* fieldprop = G4TransportationManager::GetTransportationManager()->GetPropagatorInField();
	fieldprop->SetMaxLoopCount(80) ;//check 100, 10 is bad, less is bad, default is 1000, maybe a bit less works, too


}

const std::unordered_map<int, std::unordered_map<int, std::shared_ptr<SensorContainer>>> *
B4DetectorConstruction::getIndexedSensorContainers() const {
	return &indexedSensorContainers;
}

double B4DetectorConstruction::getCaloStartZ() const {
    return m_calo_start_z;
}

G4VPhysicalVolume *B4DetectorConstruction::getWorld() const {
    return m_worldPV;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
