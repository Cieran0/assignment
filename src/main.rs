use std::{collections::HashSet, error::Error, fs::read_to_string, mem};

use glam::{Mat3, Mat4, Vec3, Vec4};
use glfw::{Action, Context, Key, OpenGlProfileHint};
use rand::Rng;

struct ObjData {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[i32; 3]>,
}

fn obj_to_opengl(data: ObjData) -> (u32, i32) {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut colours: Vec<Vec4> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();

    let mut rng = rand::rng();

    for face in &data.faces {
        let v0 = Vec3::from(data.vertices[face[0] as usize]);
        let v1 = Vec3::from(data.vertices[face[1] as usize]);
        let v2 = Vec3::from(data.vertices[face[2] as usize]);

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalize();

        for &idx in face {
            let pos = Vec3::from(data.vertices[idx as usize]);
            positions.push(pos);
            colours.push(Vec4::new(
                rng.random_range(0.1..1.0),
                rng.random_range(0.1..1.0),
                rng.random_range(0.1..1.0),
                1.0,
            ));
            normals.push(normal);
        }
    }

    let vertex_count = positions.len() as i32;
    let mut vao = 0;
    let mut vbo = [0; 3];

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(3, vbo.as_mut_ptr());
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[0]);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (positions.len() * mem::size_of::<Vec3>()) as isize,
            positions.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[1]);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (colours.len() * mem::size_of::<Vec4>()) as isize,
            colours.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
        gl::EnableVertexAttribArray(1);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[2]);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (normals.len() * mem::size_of::<Vec3>()) as isize,
            normals.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
        gl::EnableVertexAttribArray(2);
    }

    (vao, vertex_count)
}

// Create a simple emissive cube (light bulb)
fn create_light_cube() -> (u32, i32) {
    let size = 0.15;
    let vertices: Vec<Vec3> = vec![
        // Front
        Vec3::new(-size, -size,  size),
        Vec3::new( size, -size,  size),
        Vec3::new( size,  size,  size),
        Vec3::new(-size,  size,  size),
        // Back
        Vec3::new(-size, -size, -size),
        Vec3::new( size, -size, -size),
        Vec3::new( size,  size, -size),
        Vec3::new(-size,  size, -size),
    ];

    // All vertices use same solid emissive colour
    let colours: Vec<Vec4> = vec![Vec4::new(1.0, 1.0, 0.8, 1.0); vertices.len()];

    // Normals: not used when emitmode=1, but we must provide
    let normals: Vec<Vec3> = vec![Vec3::ZERO; vertices.len()];

    let indices: Vec<u32> = vec![
        0, 1, 2, 2, 3, 0,
        1, 5, 6, 6, 2, 1,
        7, 6, 5, 5, 4, 7,
        4, 0, 3, 3, 7, 4,
        4, 5, 1, 1, 0, 4,
        3, 2, 6, 6, 7, 3,
    ];

    let mut vao = 0;
    let mut vbo = [0; 3];
    let mut ebo = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(3, vbo.as_mut_ptr());
        gl::GenBuffers(1, &mut ebo);

        gl::BindVertexArray(vao);

        // Position
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[0]);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * mem::size_of::<Vec3>()) as isize,
            vertices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
        gl::EnableVertexAttribArray(0);

        // Colour (solid emissive)
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[1]);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (colours.len() * mem::size_of::<Vec4>()) as isize,
            colours.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
        gl::EnableVertexAttribArray(1);

        // Normal (dummy)
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[2]);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (normals.len() * mem::size_of::<Vec3>()) as isize,
            normals.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
        gl::EnableVertexAttribArray(2);

        // EBO
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (indices.len() * mem::size_of::<u32>()) as isize,
            indices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
    }

    (vao, indices.len() as i32)
}

fn parse_obj_file(path: &str) -> Result<ObjData, Box<dyn Error>> {
    let mut obj_data = ObjData {
        vertices: Vec::new(),
        faces: Vec::new(),
    };

    let data = read_to_string(path)
        .map_err(|e| format!("Couldn't read file {}: {}", path, e))?;

    for line in data.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let x: f32 = parts.next().ok_or("Missing x")?.parse()?;
                let y: f32 = parts.next().ok_or("Missing y")?.parse()?;
                let z: f32 = parts.next().ok_or("Missing z")?.parse()?;
                if parts.next().is_some() {
                    return Err(format!("Extra data in vertex line: {}", line).into());
                }
                obj_data.vertices.push([x, y, z]);
            }
            Some("f") => {
                let mut indices = Vec::new();
                for part in parts {
                    let idx_str = part.split('/').next().unwrap_or(part);
                    let idx: i32 = idx_str.parse::<i32>()?;
                    indices.push(idx - 1);
                }
                if indices.len() != 3 {
                    return Err(format!("Only triangular faces supported: {}", line).into());
                }
                obj_data.faces.push([indices[0], indices[1], indices[2]]);
            }
            _ => continue,
        }
    }

    Ok(obj_data)
}

fn normalize_vertices(vertices: &mut Vec<[f32; 3]>) {
    if vertices.is_empty() {
        return;
    }

    // Compute bounding box
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

    for v in vertices.iter() {
        let pos = Vec3::from(*v);
        min = min.min(pos);
        max = max.max(pos);
    }

    let center = (min + max) * 0.5;
    let size = max - min;
    let max_dim = size.x.max(size.y).max(size.z);

    // Avoid division by zero
    let scale = if max_dim > 0.0 { 2.0 / max_dim } else { 1.0 };

    // Apply transformation: move to origin, then scale to [-1,1]
    for v in vertices.iter_mut() {
        let pos = Vec3::from(*v);
        let normalized = (pos - center) * scale;
        *v = normalized.to_array();
    }
}

fn compile_shader(source: &str, shader_type: u32) -> Result<u32, String> {
    let shader = unsafe { gl::CreateShader(shader_type) };
    let c_source = std::ffi::CString::new(source).expect("Shader source must be valid UTF-8");
    unsafe {
        gl::ShaderSource(shader, 1, &c_source.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);

        let mut success: i32 = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success != i32::from(gl::TRUE) {
            let mut len: i32 = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut error = vec![0; len as usize];
            gl::GetShaderInfoLog(shader, len, std::ptr::null_mut(), error.as_mut_ptr());
            let error = std::ffi::CStr::from_ptr(error.as_ptr()).to_str().unwrap().to_string();
            return Err(format!("Shader compilation failed: {}", error));
        }
    }
    Ok(shader)
}

fn create_shader_program(vertex_src: &str, fragment_src: &str) -> Result<u32, String> {
    let vertex_shader = compile_shader(vertex_src, gl::VERTEX_SHADER)?;
    let fragment_shader = compile_shader(fragment_src, gl::FRAGMENT_SHADER)?;

    let program = unsafe { gl::CreateProgram() };
    unsafe {
        gl::AttachShader(program, vertex_shader);
        gl::AttachShader(program, fragment_shader);
        gl::LinkProgram(program);

        let mut success: i32 = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        if success != i32::from(gl::TRUE) {
            let mut len: i32 = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut error = vec![0; len as usize];
            gl::GetProgramInfoLog(program, len, std::ptr::null_mut(), error.as_mut_ptr());
            let error = std::ffi::CStr::from_ptr(error.as_ptr()).to_str().unwrap().to_string();
            return Err(format!("Shader linking failed: {}", error));
        }

        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
    }

    Ok(program)
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    up: Vec3,
}

fn create_gear(
    inner_radius: f32,
    outer_radius: f32,
    height: f32,
    tooth_depth: f32,
    num_teeth: usize, // number of teeth around the gear
) -> ObjData {
    let segments = 32;
    let mut vertices = Vec::new();
    let mut faces: Vec<[i32; 3]> = Vec::new();

    // --- Inner and outer cylinder vertices ---
    for i in 0..segments {
        let angle = (i as f32) / (segments as f32) * std::f32::consts::TAU;
        let cos = angle.cos();
        let sin = angle.sin();

        // Inner
        vertices.push([inner_radius * cos, inner_radius * sin, 0.0]); // bottom
        vertices.push([inner_radius * cos, inner_radius * sin, height]); // top

        // Outer
        vertices.push([outer_radius * cos, outer_radius * sin, 0.0]); // bottom
        vertices.push([outer_radius * cos, outer_radius * sin, height]); // top
    }

    // --- Cylinder faces ---
    for i in 0..segments {
        let next = (i + 1) % segments;

        let ibb = (i * 4) as i32;
        let inn = (next * 4) as i32;

        // Outer wall
        faces.push([ibb + 2, inn + 2, inn + 3]);
        faces.push([ibb + 2, inn + 3, ibb + 3]);

        // Inner wall (hole) - reverse winding
        faces.push([ibb + 1, inn + 1, inn]);
        faces.push([ibb + 1, inn, ibb]);
        
        // Top cap
        faces.push([ibb + 1, inn + 1, inn + 3]);
        faces.push([ibb + 1, inn + 3, ibb + 3]);

        // Bottom cap
        faces.push([ibb, inn, inn + 2]);
        faces.push([ibb, inn + 2, ibb + 2]);
    }

    // --- Teeth ---
    for t in 0..num_teeth {
        let angle_offset = (t as f32) / (num_teeth as f32) * std::f32::consts::TAU;
        let cos_a = angle_offset.cos();
        let sin_a = angle_offset.sin();

        let tooth_tip_radius = outer_radius + tooth_depth;

        // base tooth vertices (local, unrotated)
        let local = [
            [outer_radius, 0.0, 0.0],
            [outer_radius, tooth_depth, 0.0],
            [tooth_tip_radius, 0.0, 0.0],
            [outer_radius, 0.0, height],
            [outer_radius, tooth_depth, height],
            [tooth_tip_radius, 0.0, height],
        ];

        let start_idx = vertices.len() as i32;

        // rotate and add vertices
        for v in &local {
            let x = v[0] * cos_a - v[1] * sin_a;
            let y = v[0] * sin_a + v[1] * cos_a;
            let z = v[2];
            vertices.push([x, y, z]);
        }

        // add faces (same as single tooth, offset by start_idx)
        faces.push([start_idx, start_idx + 1, start_idx + 2]);
        faces.push([start_idx + 3, start_idx + 5, start_idx + 4]);

        faces.push([start_idx, start_idx + 3, start_idx + 4]);
        faces.push([start_idx, start_idx + 4, start_idx + 1]);

        faces.push([start_idx + 1, start_idx + 4, start_idx + 5]);
        faces.push([start_idx + 1, start_idx + 5, start_idx + 2]);

        faces.push([start_idx + 2, start_idx + 5, start_idx + 3]);
        faces.push([start_idx + 2, start_idx + 3, start_idx]);
    }

    ObjData { vertices, faces }
}



fn main() -> Result<(), Box<dyn Error>> {
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    const MOUSE_SENSITIVITY: f32 = 0.002;

    let mut obj_data = parse_obj_file("escape.obj")?;
    normalize_vertices(&mut obj_data.vertices);

    let gear_obj = create_gear(
        0.3,
        0.5,
        0.1,
        0.15,
        20,
    );

    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 6));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlDebugContext(true));

    let (mut window, events) = glfw
        .create_window(1280, 720, "Emissive Light Source", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window");

    window.make_current();
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_mouse_button_polling(true);
    window.set_cursor_mode(glfw::CursorMode::Disabled);

    gl::load_with(|symbol| window.get_proc_address(symbol).unwrap() as *const _);

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
    }

    let shader_program = create_shader_program(
        &read_to_string("poslight.vert")?,
        &read_to_string("poslight.frag")?
    ).map_err(|e| e.to_string())?;

    let (main_vao, main_vertex_count) = obj_to_opengl(obj_data);
    let (light_vao, light_index_count) = create_light_cube();
    let (gear_vao, gear_vertex_count) = obj_to_opengl(gear_obj);

    let mut camera = Camera {
        position: Vec3::new(0.0, 0.0, 3.0),
        yaw: -std::f32::consts::FRAC_PI_2,
        pitch: 0.0,
        up: Vec3::new(0.0, 1.0, 0.0),
    };

    let projection = Mat4::perspective_rh(
        std::f32::consts::PI / 4.0,
        1280.0 / 720.0,
        0.1,
        100.0,
    );

    let mut keys_pressed: HashSet<Key> = HashSet::new();
    let mut light_position = Vec3::new(2.0, 2.0, 2.0); // Start here
    let speed = 0.05;

    // Cache uniform locations (do once)
    let model_loc = unsafe { gl::GetUniformLocation(shader_program, b"model\0".as_ptr() as *const _) };
    let view_loc = unsafe { gl::GetUniformLocation(shader_program, b"view\0".as_ptr() as *const _) };
    let proj_loc = unsafe { gl::GetUniformLocation(shader_program, b"projection\0".as_ptr() as *const _) };
    let normal_loc = unsafe { gl::GetUniformLocation(shader_program, b"normalmatrix\0".as_ptr() as *const _) };
    let lightpos_loc = unsafe { gl::GetUniformLocation(shader_program, b"lightpos\0".as_ptr() as *const _) };
    let emitmode_loc = unsafe { gl::GetUniformLocation(shader_program, b"emitmode\0".as_ptr() as *const _) };
    let attenuationmode_loc = unsafe { gl::GetUniformLocation(shader_program, b"attenuationmode\0".as_ptr() as *const _) };

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true);
                }
                glfw::WindowEvent::Key(key, _, Action::Press, _) => {
                    keys_pressed.insert(key);
                }
                glfw::WindowEvent::Key(key, _, Action::Release, _) => {
                    keys_pressed.remove(&key);
                }
                glfw::WindowEvent::CursorPos(xpos, ypos) => {
                    if mouse_pressed {
                        if let Some((last_x, last_y)) = last_mouse_pos {
                            let dx = (xpos - last_x) as f32;
                            let dy = (ypos - last_y) as f32;
                            camera.yaw += dx * MOUSE_SENSITIVITY;
                            camera.pitch -= dy * MOUSE_SENSITIVITY;
                            camera.pitch = camera.pitch.clamp(
                                -std::f32::consts::FRAC_PI_2 + 0.01,
                                std::f32::consts::FRAC_PI_2 - 0.01,
                            );
                        }
                        last_mouse_pos = Some((xpos, ypos));
                    } else {
                        last_mouse_pos = Some((xpos, ypos));
                    }
                }
                glfw::WindowEvent::MouseButton(glfw::MouseButton::Left, action, _) => {
                    mouse_pressed = action == Action::Press;
                    if mouse_pressed {
                        let (x, y) = window.get_cursor_pos();
                        last_mouse_pos = Some((x, y));
                    }
                }
                _ => {}
            }
        }

        // Camera movement
        let forward = Vec3::new(
            camera.yaw.cos() * camera.pitch.cos(),
            camera.pitch.sin(),
            camera.yaw.sin() * camera.pitch.cos(),
        ).normalize();
        let right = forward.cross(camera.up).normalize();
        let world_up = Vec3::new(0.0, 1.0, 0.0);

        if keys_pressed.contains(&Key::W) { camera.position += forward * speed; }
        if keys_pressed.contains(&Key::S) { camera.position -= forward * speed; }
        if keys_pressed.contains(&Key::A) { camera.position -= right * speed; }
        if keys_pressed.contains(&Key::D) { camera.position += right * speed; }
        if keys_pressed.contains(&Key::Space) { camera.position += world_up * speed; }
        if keys_pressed.contains(&Key::LeftShift) { camera.position -= world_up * speed; }

        // Light movement
        if keys_pressed.contains(&Key::Up) { light_position.z -= speed; }
        if keys_pressed.contains(&Key::Down) { light_position.z += speed; }
        if keys_pressed.contains(&Key::Left) { light_position.x -= speed; }
        if keys_pressed.contains(&Key::Right) { light_position.x += speed; }
        if keys_pressed.contains(&Key::PageUp) { light_position.y += speed; }
        if keys_pressed.contains(&Key::PageDown) { light_position.y -= speed; }

        let target = camera.position + forward;
        let view = Mat4::look_at_rh(camera.position, target, camera.up);
        let light_view = (view * light_position.extend(1.0)).truncate();

        unsafe {
            gl::ClearColor(0.05, 0.05, 0.05, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            gl::UseProgram(shader_program);
            gl::UniformMatrix4fv(proj_loc, 1, gl::FALSE, projection.to_cols_array().as_ptr());
            gl::UniformMatrix4fv(view_loc, 1, gl::FALSE, view.to_cols_array().as_ptr());
            gl::Uniform4f(lightpos_loc, light_view.x, light_view.y, light_view.z, 1.0);
            gl::Uniform1ui(attenuationmode_loc, 1);

            // === Render main object (non-emissive) ===
            let model = Mat4::IDENTITY;
            let normal_matrix = Mat3::from_mat4(model).inverse().transpose();
            gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.to_cols_array().as_ptr());
            gl::UniformMatrix3fv(normal_loc, 1, gl::FALSE, normal_matrix.to_cols_array().as_ptr());
            gl::Uniform1ui(emitmode_loc, 0); // ← not emissive

            gl::BindVertexArray(main_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, main_vertex_count);

            // === Render light object (emissive) ===
            let light_model = Mat4::from_translation(light_position);
            // For light object, normal matrix doesn't matter (emitmode=1 ignores lighting)
            gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, light_model.to_cols_array().as_ptr());
            gl::UniformMatrix3fv(normal_loc, 1, gl::FALSE, Mat3::IDENTITY.to_cols_array().as_ptr());
            gl::Uniform1ui(emitmode_loc, 1); // ← emissive!

            gl::BindVertexArray(light_vao);
            gl::DrawElements(gl::TRIANGLES, light_index_count, gl::UNSIGNED_INT, std::ptr::null());

            let gear_model = Mat4::from_rotation_z(0.0) * Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0));
            let gear_normal_matrix = Mat3::from_mat4(gear_model).inverse().transpose();
            gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, gear_model.to_cols_array().as_ptr());
            gl::UniformMatrix3fv(normal_loc, 1, gl::FALSE, gear_normal_matrix.to_cols_array().as_ptr());
            gl::Uniform1ui(emitmode_loc, 0);

            gl::BindVertexArray(gear_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, gear_vertex_count);
        }

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteProgram(shader_program);
    }

    Ok(())
}