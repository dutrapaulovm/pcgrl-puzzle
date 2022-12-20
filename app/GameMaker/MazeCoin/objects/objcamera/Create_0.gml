/// @description Insert description here
target = noone;

snapDiv = 16; 

leave_room = true

r_width  = camera_get_view_width(view_camera[0]);
r_height = camera_get_view_width(view_camera[0]);

v_wport = camera_get_view_width(view_camera[0]);
v_hport = camera_get_view_width(view_camera[0]);

m_min_border_x = 0;
m_max_border_x = camera_get_view_width(view_camera[0]);
m_min_border_y = 0;
m_max_border_y = camera_get_view_height(view_camera[0]);

hb =  round(camera_get_view_width(view_camera[0]) / 2)-16
vb =  round(camera_get_view_height(view_camera[0]) / 2)-16

camera_set_view_target(view_camera[0], id);
camera_set_view_border(view_camera[0],hb,vb)