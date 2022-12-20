/// @description Lógica da camera
	instance_activate_object(objCamera);
if (target != noone)
{
	
	x = target.x;
	y = target.y;
	
//	var b = 32 * target.flip
	
//	var x_follow = target.x;
//	var y_follow = target.y;
    
/*	if (!leave_room) {
	    x_follow = clamp(x_follow, 0, r_width-1); 
	    y_follow = clamp(y_follow, 0, r_height-1);   
	}*/
	
	//x_follow = floor(x_follow/v_wport)*v_wport;
	//y_follow = floor(y_follow/v_hport)*v_hport;
	
	//camera_set_view_pos(view_camera[0],x_follow,y_follow);

	
    //x += (target.x - x) / snapDiv;
    //y += (target.y - y) / snapDiv;	

	
	/*if (target.x+16 > m_max_border_x)
	{		
		m_min_border_x = m_max_border_x;
		m_max_border_x += camera_get_view_width(view_camera[0]);
		camera_set_view_pos(view_camera[0], m_max_border_x ,  m_min_border_y);		
	}
	
	if (target.x < m_min_border_x)
	{		
		m_min_border_x -= camera_get_view_width(view_camera[0]);
		m_max_border_x -= camera_get_view_width(view_camera[0]);	
		camera_set_view_pos(view_camera[0], m_min_border_x, m_min_border_y);
	}
	
	if (target.y+16 > m_max_border_y)
	{		
		m_min_border_y = m_max_border_y;
		m_max_border_y += camera_get_view_height(view_camera[0]);
		camera_set_view_pos(view_camera[0],  m_min_border_x, m_min_border_y);
	}
	
	if (target.y < m_min_border_y)
	{		
		m_min_border_y -= camera_get_view_height(view_camera[0]);
		m_max_border_y -= camera_get_view_height(view_camera[0]);	
		camera_set_view_pos(view_camera[0], m_min_border_x, m_min_border_y);
	}
	
	//Ativa somente as instancias dentro de uma área
	instance_deactivate_all(true);
	//camera_set_view_border(view_camera[0],16,16)
	var _vx = camera_get_view_x(view_camera[0]);
	var _vy = camera_get_view_y(view_camera[0]);
	var _vw = camera_get_view_width(view_camera[0]);
	var _vh = camera_get_view_height(view_camera[0]);
	instance_activate_region(_vx, _vy, _vw, _vh , false);	
	instance_activate_object(objInit);
	instance_activate_object(objCamera);	
	instance_activate_object(objController);	
	instance_activate_object(objPlayer);	
	instance_activate_object(objSolids);
	instance_activate_object(objGold);*/
}	