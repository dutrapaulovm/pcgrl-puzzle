// Script assets have changed for v2.3.0 see
// https://help.yoyogames.com/hc/en-us/articles/360005277377 for more information
function next_level(){
	global.map_id += 1
	if (global.map_id > global.max_levels)
		room_goto(rmThanks)
	else{
		global.map_id = min(global.max_levels, max(1, global.map_id))	
		room_restart()
	}
}