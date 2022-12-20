/// @description Insert description here
// You can write your code in this editor
with(other)
	instance_destroy()

golds += 1
audio_play_sound(sfx_gold, 10, false);

if (instance_number(objGold) == 0){
	next_level()
}


