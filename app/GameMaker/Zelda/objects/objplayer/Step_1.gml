/// @description Insert description here
// You can write your code in this editor
var kleft   = false;
var kright  = false;

kleft  = keyboard_check(vk_left);
kright = keyboard_check(vk_right);	

kup    = keyboard_check(vk_up);
kdown  = keyboard_check(vk_down);

var wleft   = place_meeting(x-1,y, objSolids);
var wright  = place_meeting(x+1,y, objSolids);	

var wabove  = place_meeting(x,y-1, objSolids);
var wbelow  = place_meeting(x,y+1, objSolids);	

//Calcula a direção do paddle para ser utilizada no calculo do movimento no eixo X
var dir_x = kright - kleft;
var dir_y = kdown  - kup;

var move_x = 0;
var move_y = 0;

m_state = STATE_PLAYER.idle;

if (kleft && !kright)
{
	move_x = (m_move_spd * delta_time * MS_TO_S60) * dir_x;
	flip = -1
	m_state = STATE_PLAYER.run
}
else
	if (!kleft && kright) 
	{
		move_x = (m_move_spd * delta_time * MS_TO_S60) *  dir_x;
		flip = 1
		m_state = STATE_PLAYER.run
	}

if (kup && !kdown)
{
	move_y = (m_move_spd * delta_time * MS_TO_S60) * dir_y;
	m_state = STATE_PLAYER.run
}
else
	if (!kup && kdown) 
	{
		move_y = (m_move_spd  * delta_time * MS_TO_S60) * dir_y;
		m_state = STATE_PLAYER.run
	}

repeat(m_move_spd){
	if(!place_meeting(x+sign(move_x), y, objSolids)){
		x += sign(move_x);
	}
	else{
		move_x= 0;
	}
}	

repeat(m_move_spd){
	if(!place_meeting(x, y+sign(move_y), objSolids)){
		y += sign(move_y);
	}
	else{
		move_y= 0;
	}
}

switch(m_state){
	
	case STATE_PLAYER.idle:
		sprite_index = sprPlayerIdle
	break
	case STATE_PLAYER.run:
		sprite_index = sprPlayerRun	
		start_timer()		
	break	
	
}

