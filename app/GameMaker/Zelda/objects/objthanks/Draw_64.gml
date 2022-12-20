draw_set_font(fntEnd);
draw_set_halign(fa_center);
draw_set_valign(fa_center);

var xx = 16;
var yy = 16;
//var line_spacing = 16
//yy += line_spacing
draw_set_alpha(0.5);
draw_set_colour(c_black);
draw_rectangle_colour(4, 4 , room_width-4, room_height-4, c_black, c_black, c_black, c_black, false);
draw_set_alpha(1);
draw_set_colour(c_white);
draw_text(round(room_width/2), round(room_height/2), "Thank you for playing");

