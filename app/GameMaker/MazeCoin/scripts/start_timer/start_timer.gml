// Script assets have changed for v2.3.0 see
// https://help.yoyogames.com/hc/en-us/articles/360005277377 for more information
function start_timer(){
if !go{

			if (instance_exists(objController)){         
				with(objController)
				{
				    alarm[0] = 60
				    alarm[1] = 1
				}
			}  
	
			go = true;

		}		
}