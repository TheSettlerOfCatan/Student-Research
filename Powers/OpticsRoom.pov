#version 3.8;
global_settings {
	assumed_gamma 1
	radiosity {
		pretrace_start 0.08
		pretrace_end   0.04
		count 35

		nearest_count 5
		error_bound 1.8
		recursion_limit 3

		low_error_factor 0.5
		gray_threshold 0.0
		minimum_reuse 0.015
		brightness 1

		adc_bailout 0.01/2
	}
}
#include "colors.inc"
#include "textures.inc"

/************************************************* room w/ doors *************************************************/

#declare room_z = 234;
#declare room_x = 237.5;
#declare room_height = 108;

union {
	box { <0,0,0>, <room_x, room_height*10, room_z> }
	box { <room_x-12, 0, room_z-12>, <room_x, room_height, room_z> }
}


box { <17.5, 0, room_z>, <62.5, 86, room_z-0.5> }
box { <19.5, 0, room_z>, <60.5, 84, room_z-0.5> 
	texture { pigment { color Copper } }
}
union { // handle
	cylinder {<58, 46, room_z-0.5>, <58, 46, room_z-3.5>, 0.5
	finish { Metal }
	}
	box {<58, 45.5, room_z-3.5>, <54, 46.5, room_z-3.2>
	finish { Metal }
	}
}

box { <room_x, 0, 62>, <room_x-0.5, 86, 107> }
box { <room_x, 0, 64>, <room_x-0.5, 84, 105> 
	texture { pigment { color Copper } }
}
union { // handle
	box {<room_x-0.5, 44.5, 66>, <room_x-2, 47.5, 103>
	finish { Metal }
	}
	box {<room_x-2, 45, 67.5>, <room_x-3.5, 47, 101.5>
	finish { Metal }
	}
	box {<room_x-3.5, 44.3, 66>, <room_x-4, 47.7, 103>
	finish { Metal }
	}
}

//CEILING
//plane

/************************************************* camera *************************************************/

/*
camera {
//	orthographic
	location <room_x/2, room_height*10, room_z/2>
	look_at <room_x/2, room_height/2, room_z/2>
	angle 18
}
*/
camera {
	location <6, 74, 0>
	look_at <room_x/2, room_height/2, room_z/2>
	angle 90
}
/*
camera {
	orthographic
	location <room_x/2, room_height, room_z/2>
	look_at <room_x/2, 0, room_z/2>
	angle 110
}*/

/************************************************* lights *************************************************/

light_source {
	<room_x/2, room_height-1, room_z/2>
	color White
	area_light <room_x/2, 0, 0>, <0, 0, room_z/2>, 20, 20
	adaptive 1
	jitter
}

/************************************************* optical tables *************************************************/

#declare table_length = 96;
#declare table_width = 49.5;
#declare table1_corner_x = 50;
#declare table1_corner_z = 90;
#declare table2_corner_x = 150;
#declare table2_corner_z = 90;
#declare table_bottom_height = 29;
#declare table_top_height = 41;

union {
	box{ <table1_corner_x,table_bottom_height,table1_corner_z>,<table1_corner_x+table_width,table_top_height,table1_corner_z+table_length> 
		pigment { granite color_map { [0 rgb<0.2,0.3,0.3>*0.3]
									  [1 rgb<0.2,0.3,0.3>*0.2]
									} 
				  scale 2
				}                              
		finish { ambient 0 diffuse 1.0 specular 0.1 roughness 0.1 brilliance 1 phong 0.2 phong_size 200 }
		normal { granite 0.21 scale 2}
	}
	cylinder { <table1_corner_x+8, 0, table1_corner_z+8>, <table1_corner_x+8, table_bottom_height, table1_corner_z+8>, 7
	texture { pigment { color Black } }
	}
	cylinder { <table1_corner_x+table_width-8, 0, table1_corner_z+8>, <table1_corner_x+table_width-8, table_bottom_height, table1_corner_z+8>, 7
	texture { pigment { color Black } }
	}
	cylinder { <table1_corner_x+8, 0, table1_corner_z+table_length-8>, <table1_corner_x+8, table_bottom_height, table1_corner_z+table_length-8>, 7
	texture { pigment { color Black } }
	}
	cylinder { <table1_corner_x+table_width-8, 0, table1_corner_z+table_length-8>, <table1_corner_x+table_width-8, table_bottom_height, table1_corner_z+table_length-8>, 7
	texture { pigment { color Black } }
	}
}

union {
	box{ <table2_corner_x,table_bottom_height,table2_corner_z>,<table2_corner_x+table_width,table_top_height,table2_corner_z+table_length> 
		pigment { granite color_map { [0 rgb<0.2,0.3,0.3>*0.3]
									  [1 rgb<0.2,0.3,0.3>*0.2]
									} 
				  scale 2
				}                              
		finish { ambient 0 diffuse 1.0 specular 0.1 roughness 0.1 brilliance 1 phong 0.2 phong_size 200 }
		normal { granite 0.21 scale 2}
	}
	cylinder { <table2_corner_x+8, 0, table2_corner_z+8>, <table2_corner_x+8, table_bottom_height, table2_corner_z+8>, 7
	texture { pigment { color Black } }
	}
	cylinder { <table2_corner_x+table_width-8, 0, table2_corner_z+8>, <table2_corner_x+table_width-8, table_bottom_height, table2_corner_z+8>, 7
	texture { pigment { color Black } }
	}
	cylinder { <table2_corner_x+8, 0, table2_corner_z+table_length-8>, <table2_corner_x+8, table_bottom_height, table2_corner_z+table_length-8>, 7
	texture { pigment { color Black } }
	}
	cylinder { <table2_corner_x+table_width-8, 0, table2_corner_z+table_length-8>, <table2_corner_x+table_width-8, table_bottom_height, table2_corner_z+table_length-8>, 7
	texture { pigment { color Black } }
	}
}

/************************************************* cabinets *************************************************/

box { <room_x-74, 2*room_height/3-2, room_z-17.5>, <room_x-12, room_height-2, room_z>
	texture { pigment { color Copper } }
}
union {
	box { <room_x-73, 2*room_height/3-1, room_z-17.5>, <room_x-43.2, room_height-3, room_z-17.8>
	texture { pigment { color Copper } }	
	}
	cylinder { <room_x-45, 2*room_height/3+2, room_z-17.8>, <room_x-45, 2*room_height/3+2, room_z-19>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-45, 2*room_height/3+5, room_z-17.8>, <room_x-45, 2*room_height/3+5, room_z-19>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-45, 2*room_height/3+1, room_z-19>, <room_x-45, 2*room_height/3+6, room_z-19>, 0.2
	finish { Metal }
	}
}
union {
	box { <room_x-42.8, 2*room_height/3-1, room_z-17.5>, <room_x-13, room_height-3, room_z-17.8>
	texture { pigment { color Copper } }	
	}
	cylinder { <room_x-41, 2*room_height/3+2, room_z-17.8>, <room_x-41, 2*room_height/3+2, room_z-19>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-41, 2*room_height/3+5, room_z-17.8>, <room_x-41, 2*room_height/3+5, room_z-19>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-41, 2*room_height/3+1, room_z-19>, <room_x-41, 2*room_height/3+6, room_z-19>, 0.2
	finish { Metal }
	}
}


box { <room_x-24, 2*room_height/3-2, 0>, <room_x, room_height-2, 62>
	texture { pigment { color Copper } }
}
union {
	box { <room_x-24, 2*room_height/3-1, 1>, <room_x-24.3, room_height-3, 30.8> 
	texture { pigment { color Copper } }
	}
	cylinder { <room_x-24.3, 2*room_height/3+2, 29>, <room_x-25.5, 2*room_height/3+2, 29>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-24.3, 2*room_height/3+5, 29>, <room_x-25.5, 2*room_height/3+5, 29>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-25.5, 2*room_height/3+1, 29>, <room_x-25.5, 2*room_height/3+6, 29>, 0.2
	finish { Metal }
	}
}
union {
	box { <room_x-24, 2*room_height/3-1, 31.2>, <room_x-24.3, room_height-3, 61> 
	texture { pigment { color Copper } }
	}
	cylinder { <room_x-24.3, 2*room_height/3+2, 33>, <room_x-25.5, 2*room_height/3+2, 33>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-24.3, 2*room_height/3+5, 33>, <room_x-25.5, 2*room_height/3+5, 33>, 0.2
	finish { Metal }
	}
	cylinder { <room_x-25.5, 2*room_height/3+1, 33>, <room_x-25.5, 2*room_height/3+6, 33>, 0.2
	finish { Metal }
	}
}


union {
	box { <room_x-24, 41, 62>, <room_x, 40, 0>
		texture { pigment { color Black } }
	}
	box { <room_x-22.5, 40, 60.5>, <room_x, 3, 0>
		texture { pigment { color Copper } }
	}
	box { <room_x-20.5, 3, 60.5>, <room_x, 0, 0>
		texture { pigment { color Copper } }
	}
}
union {
	box { <room_x-22.5, 39, 59.5>, <room_x-22.8, 4, 30.45>
		texture { pigment { color Copper } }
	}
	cylinder { <room_x-22.8, 23, 32.25>, <room_x-24, 23, 32.25>, 0.2
		finish { Metal }
	}
	cylinder { <room_x-22.8, 20, 32.25>, <room_x-24, 20, 32.25>, 0.2
		finish { Metal }
	}	
	cylinder { <room_x-24, 19, 32.25>, <room_x-24, 24, 32.25>, 0.2
		finish { Metal }
	}
}
union {
	box { <room_x-22.5, 39, 30.05>, <room_x-22.8, 4, 1>
		texture { pigment { color Copper } }
	}
	cylinder { <room_x-22.8, 23, 28.25>, <room_x-24, 23, 28.25>, 0.2
		finish { Metal }
	}
	cylinder { <room_x-22.8, 20, 28.25>, <room_x-24, 20, 28.25>, 0.2
		finish { Metal }
	}	
	cylinder { <room_x-24, 19, 28.25>, <room_x-24, 24, 28.25>, 0.2
		finish { Metal }
	}
}

/************************************************* other stuff *************************************************/

//TABLE
union {
	box { <room_x-74.2, 41, room_z-30.5>, <room_x-12.2, 39.3, room_z>
		texture { pigment { color Copper } }
	}
	cone { <room_x-16.2, 0, room_z-4>, 1, <room_x-16.2, 39.3, room_z-4>, 1.5
		texture { pigment { color Copper } }
	}
	cone { <room_x-16.2, 0, room_z-26.5>, 1, <room_x-16.2, 39.3, room_z-26.5>, 1.5
		texture { pigment { color Copper } }
	}
	cone { <room_x-70.2, 0, room_z-4>, 1, <room_x-70.2, 39.3, room_z-4>, 1.5
		texture { pigment { color Copper } }
	}
	cone { <room_x-70.2, 0, room_z-26.5>, 1, <room_x-70.2, 39.3, room_z-26.5>, 1.5
		texture { pigment { color Copper } }
	}	
}

//RANDOM PIPE
cylinder { <2.6, 0, 95.5>, <2.6, room_height, 95.5>, 2.5 }











