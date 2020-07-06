#version 3.8;
global_settings{ assumed_gamma 1.0 }

#include "colors.inc"
#include "textures.inc"

camera {
	orthographic
	location <0,10,0>
	look_at <0,0,0>
}

#declare theta1 = theta*pi/180;
#declare phi1 = phi*pi/180;

light_source{<cos(theta1)*sin(phi1),cos(phi1),sin(theta1)*sin(phi1)>*1000000 color White}
light_source{<-cos(theta1)*sin(phi1),cos(phi1),-sin(theta1)*sin(phi1)>*1000000 color White}

plane{ <0,1,0>, 0 
	texture{ pigment { color rgb <1,1,1> }
	}
}

cylinder{<0,0,0>, <0,1.5,0>, 1
	texture{ pigment { color rgb <0,0,1> }
	}
}
