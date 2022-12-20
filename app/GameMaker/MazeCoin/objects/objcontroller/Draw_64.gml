draw_set_font(fntHUD);
draw_set_halign(fa_left);
draw_set_valign(fa_center);

var xx = 16;
var yy = 16;
/*var line_spacing = 16
draw_text(xx, yy, string(room_width) + " x " + string(room_height));
yy += line_spacing
draw_text(xx, yy, string(global.map_id));
yy += line_spacing
draw_text(xx, yy, string(objPlayer.x) + " x " + string(objPlayer.y));
yy += line_spacing
draw_text(xx, yy, string(objCamera.x) + " x " + string(objCamera.y));
yy += line_spacing*/
draw_set_alpha(0.5);
draw_set_colour(c_black);
draw_rectangle_colour(4, 4,164, 24, c_black, c_black, c_black, c_black, false);

draw_set_alpha(1);

draw_set_color(c_white)
draw_sprite_ext(spr_timer_hud, 0, xx , 16, image_xscale, image_yscale, image_angle, image_blend, image_alpha);
draw_text(xx+8, 16, string_hash_to_newline(string_add_zeros(seconds,2)+":"+string_add_zeros(miliseconds,2)));

draw_sprite_ext(sprGold, 0, xx+48 , 8, image_xscale, image_yscale, image_angle, image_blend, image_alpha);
draw_text(xx+64, 16, string_hash_to_newline(string_add_zeros(objPlayer.golds,2)+"/"+string_add_zeros(golds_counts, 2)));

draw_text(xx+100, 16, "Lv: " + string_hash_to_newline(string_add_zeros(global.map_id,2)+"/"+string_add_zeros(global.max_levels, 2)));


