#version 3.8;
global_settings{ assumed_gamma 1.0 }

#include "colors.inc"
#include "textures.inc"

camera {
	orthographic
	location <0,10,0>
	look_at <0,0,0>
}

light_source{<-0.433012701892,0.5,-0.75>*100000 color White}
light_source{<0.433012701892,0.5,0.75>*100000 color White}

plane{ <0,1,0>, 0 
	texture{ pigment { color White }
	}
}

cylinder{<0,0,0>, <0,1.5,0>, 1
	texture{ pigment { color Blue }
	}
}
