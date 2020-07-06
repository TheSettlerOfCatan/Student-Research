$fn=300;
union() {
    difference() {
        cylinder(7.62,r=50.8/2.0,center=true);
        union() {
            cylinder(7.63,r=46.0/2.0,center=true);
            cylinder(0.955,r=50.771/2.0,center=true);
        }
    }
};
//IDM=2in=50.8mm
//IIDM=1.94in=49.3mm
//ODBS=1.9985in=50.7619mm
//hBS=.0375in=0.9525mm
//hM=0.3in=7.62mm

//M is Mount, BS is Beamsplitter, ID and OD are inner and outer diameters