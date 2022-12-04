/*
 * sensorContainer.h
 *
 *  Created on: 4 Apr 2018
 *      Author: jkiesele
 */

#ifndef B4A_INCLUDE_SENSORCONTAINER_H_
#define B4A_INCLUDE_SENSORCONTAINER_H_

#include "G4VPhysicalVolume.hh"


class SensorContainer{
public:
	SensorContainer(G4VPhysicalVolume *vol, G4double eta, G4double phi, G4double null, G4double posx,
                    G4double posy, G4double posz, G4double pre_absorber_thickness, int layer,
                    G4int copyno, G4double thickness, G4double dphi, G4double deta);


	const G4double& getArea() const {
		return area_;
	}

	const G4double& getEta() const {
		return eta_;
	}

	const G4double& getPhi() const {
		return phi_;
	}

	const G4VPhysicalVolume* getVol() const {
		return vol_;
	}

	const G4double& getPosx() const {
		return posx_;
	}

	const G4double& getPosy() const {
		return posy_;
	}

	const G4double& getPosz() const {
		return posz_;
	}

	G4double getEnergyscalefactor() const {
		return energyscalefactor_;
	}

	void setEnergyscalefactor(G4double energyscalefactor) {
		energyscalefactor_ = energyscalefactor;
	}

    G4double getPreAbsorberThickness() const;

    const int& getLayer() const {
		return layer_;
	}

	G4int getCopyNo()const{
	    return copyno_;
	}

	const int& getGlobalDetID()const{
		return global_detid_;
	}

private:
	SensorContainer(){
			global_detid_=global_detid_counter_++;
		}

public:
    G4double getThickness() const;

private:

    G4VPhysicalVolume * vol_;
	G4double eta_;
	G4double phi_;
	G4double area_;

	G4double posx_;
	G4double posy_;
	G4double posz_;

    G4double deta;
    G4double dphi;
public:
    G4double getDeta() const;

    G4double getDphi() const;

private:
    G4double energyscalefactor_;

	int layer_;
	int rechit_idx_;

	int global_detid_;

	G4int  copyno_;
    G4double thickness;
    G4double pre_absorber_thickness;

	static int global_detid_counter_;

};


#endif /* B4A_INCLUDE_SENSORCONTAINER_H_ */
