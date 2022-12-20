/// @description Increment the miliseconds variable
timer++
miliseconds++
                
alarm[1] = 1;
            
if (miliseconds >= 60){
    miliseconds = 0;
}
       
	   
if (seconds == level_time){
	miliseconds = 0;
	seconds     = level_time;  
	alarm[0] = 0;
	alarm[1] = 0;
	game_restart()
}
	      
