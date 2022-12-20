// Script assets have changed for v2.3.0 see
// https://help.yoyogames.com/hc/en-us/articles/360005277377 for more information
function enconde_template() {
	grid_size = 16;
	levelData = "";
	xTiles = room_width / grid_size;
	yTiles = ceil(room_height / grid_size);
	show_debug_message(string(xTiles))
	show_debug_message(string(yTiles))
	section_list = ds_list_create();
	var layerid = layer_get_id("Tiles_1")
	var layermapid = layer_tilemap_get_id(layerid)
	section = "LEVEL";		
	//ini_open(working_directory+filename);		
	ini_open(working_directory+filename);		
		
		for (yy = 0; yy < yTiles; yy++){
		
			for ( xx= 0; xx < xTiles; xx++){
		
				 inst = instance_position(xx * grid_size, yy * grid_size, all)		
				 
				 levelData += string(tilemap_get_at_pixel(layermapid, xx * grid_size, yy * grid_size))
				 levelData += ","	
				 
				 /*if (inst != noone)
					inst = inst.object_index;
				
				 switch(inst){
			 
						case noone:
							levelData += "0"
						break;
						case objSolid01:
							levelData += "1"
						break;
						case objTemplateBlockAir:
							levelData += "6"
						break;													
						case objTemplateBlockGround:
							levelData += "5"
						break;								
						case objTemplateDoorEnter:
							levelData += "8"
						break;							
				 }*/
			 
			}
			levelData += "\n"	
		
		}
	
		show_debug_message(section + " " + string(global.current) + "  : " + levelData);
		show_debug_message("");
	
		ds_list_add(section_list, levelData);	
		ini_write_string(section, string(global.current), levelData);
		ini_write_real(section, "size", global.current);
		ini_close();
		ds_list_destroy(section_list);
		global.current += 1

}
