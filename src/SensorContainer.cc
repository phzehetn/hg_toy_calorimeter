#include "../include/SensorContainer.h"


int SensorContainer::global_detid_counter_ = 0;

G4double SensorContainer::getPreAbsorberThickness() const {
    return pre_absorber_thickness;
}


SensorContainer::SensorContainer(G4VPhysicalVolume *vol, G4double eta, G4double phi, G4double null, G4double posx,
                                 G4double posy, G4double posz, G4double pre_absorber_thickness, int layer,
                                 G4int copyno, G4double thickness, G4double dphi, G4double deta) :
        vol_(vol), eta_(eta), phi_(phi), area_(null),
        posx_(posx), posy_(posy), posz_(posz), energyscalefactor_(1),
        layer_(layer), copyno_(copyno), thickness(thickness), pre_absorber_thickness(pre_absorber_thickness),
        dphi(dphi), deta(deta) {
    global_detid_ = global_detid_counter_++;
}

G4double SensorContainer::getThickness() const {
    return thickness;
}

G4double SensorContainer::getDeta() const {
    return deta;
}

G4double SensorContainer::getDphi() const {
    return dphi;
}
