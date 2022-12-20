m_grid = 16
m_move_spd = 1
flip = 1
golds = 0

enum STATE_PLAYER
{
    idle,
    run
}

m_state = STATE_PLAYER.idle

/// @description Insert description here
// You can write your code in this editor
with(instance_create_layer(0, 0, "Controllers", objCamera))
{
	target = other;	
}

go = false;