/// @description Insert description here
file = load_csv("Map"+string(global.map_id)+".csv");
var ww = ds_grid_width(file);
var hh = ds_grid_height(file);
var xx = 0;
var yy = 0;
xx_space = 32
yy_space = 16
for (var i = 0; i < ww; i++;)
{
    for (var j = 0; j < hh; j++;)
    {			
        yy += 16;
        //draw_text(xx, yy, string(file[# i, j]));
		if file[# i, j] == 1
			instance_create_layer(xx+xx_space, yy+yy_space, "Level",objWall )
		else
		if file[# i, j] == 2
			instance_create_layer(xx+xx_space, yy+yy_space, "Items", objGold )			
		else
		if file[# i, j] == 3
			instance_create_layer(xx+xx_space, yy+yy_space, "Player",objPlayer )								
    }
    yy = 16;
    xx += 16;
}

display_set_gui_size(camera_get_view_width(view_camera[0]), camera_get_view_height(view_camera[0]))

golds_counts = instance_number(objGold)

if (instance_exists(objCamera)){
	with(objCamera){
		target = objPlayer
		//event_user(0)		
	}	
}

room_set_width(room,  (ww * 16) + 128)
room_set_height(room,  (hh * 16) + 128)