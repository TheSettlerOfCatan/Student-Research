#version 3.8;
global_settings{ assumed_gamma 1.0 }

//#include "colors.inc"
//#include "textures.inc"
//#include "math.inc"

#declare l = 0.00532;
#declare Angle = 0.006;
#declare Camera_Distance = 20000;
#declare deltakm = 1/(Camera_Distance*tan(Angle*pi/180));
#declare deltakn = 1/(Camera_Distance*tan(Angle*pi/180));
#declare theta = asin(l*sqrt(pow(m*deltakm,2)+pow(n*deltakn,2))/2);
#declare phi = atan2(Nrow/2-m,n-Ncol/2);

#debug concat("\n", str(theta*180/pi,1,3), "\n\n")

camera {
	orthographic
	location <0,Camera_Distance,0>
	look_at <0,0,0>
	sky <0,0,1>
	angle Angle
}

light_source{<-cos(phi)*sin(theta),cos(theta),-sin(phi)*sin(theta)>*100000 color rgb 1 }
light_source{<cos(phi)*sin(theta),cos(theta),sin(phi)*sin(theta)>*100000 color rgb 1 }


plane{ <0,1,0>, 0
	texture{ pigment { color rgb 1 }
		finish { ambient 0 diffuse 1 }
	}
}

cylinder{<0,0,0>, <0,0.1,0>, 0.1
	texture{ pigment { color rgb <0,0,1> }
		finish { ambient 0 diffuse 1 }
	}
}







//ArcSin[(532*(10^-9))*Sqrt[(800/(Tan[45*Pi/180]*(7.9^-4)))^2+(600/(Tan[45*Pi/180]*(7.9^-4)*(4/3)))^2]/2]*180/Pi

//ArcSin[(532*(10^-9))*Sqrt[(128/(Tan[0.08*Pi/180]*2))^2+(128/(Tan[0.08*Pi/180]*2))^2]/2]*180/Pi

//declare theta = asin(l*sqrt(pow(m*deltakm,2)+pow(n*deltakn,2))/2);
//declare phi = atan2(Nrow/2-m,n-Ncol/2);

