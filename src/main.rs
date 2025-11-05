use std::{collections::HashSet, error::Error, fs::read_to_string, mem};

use glam::{Mat3, Mat4, Vec3, Vec4};
use glfw::{Action, Context, Key, OpenGlProfileHint};

// --- TRANSFORMATION STACK ---
#[derive(Default)]
struct TransformStack {
    stack: Vec<Mat4>,
}

impl TransformStack {
    fn new() -> Self {
        Self {
            stack: vec![Mat4::IDENTITY],
        }
    }

    fn push(&mut self, transform: Mat4) {
        let current = *self.stack.last().unwrap();
        self.stack.push(current * transform);
    }

    fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }

    fn current(&self) -> Mat4 {
        *self.stack.last().unwrap()
    }
}

// --- CREATE A BOX (indexed) ---
fn create_box(width: f32, height: f32, depth: f32) -> (Vec<Vec3>, Vec<Vec4>, Vec<Vec3>, Vec<u32>) {
    let w = width / 2.0;
    let h = height / 2.0;
    let d = depth / 2.0;

    let vertices = vec![
        // Front
        Vec3::new(-w, -h,  d),
        Vec3::new( w, -h,  d),
        Vec3::new( w,  h,  d),
        Vec3::new(-w,  h,  d),
        // Back
        Vec3::new(-w, -h, -d),
        Vec3::new( w, -h, -d),
        Vec3::new( w,  h, -d),
        Vec3::new(-w,  h, -d),
    ];

    let indices = vec![
        0, 1, 2, 2, 3, 0,
        1, 5, 6, 6, 2, 1,
        7, 6, 5, 5, 4, 7,
        4, 0, 3, 3, 7, 4,
        4, 5, 1, 1, 0, 4,
        3, 2, 6, 6, 7, 3,
    ];

    let colours = vec![Vec4::new(0.6, 0.6, 0.6, 1.0); vertices.len()];
    let normals = vec![Vec3::ZERO; vertices.len()];

    (vertices, colours, normals, indices)
}

// --- CREATE PROJECTILE (cylinder + hemispherical nose) ---
fn create_projectile(radius: f32, cyl_height: f32, hemi_segments: usize) -> (Vec<Vec3>, Vec<Vec4>, Vec<Vec3>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut colours = Vec::new();
    let mut normals = Vec::new();

    // Cylinder sides
    let cyl_segments = 16;
    let cyl_side_start = vertices.len() as u32;
    for i in 0..cyl_segments {
        let angle1 = (i as f32) * 2.0 * std::f32::consts::PI / (cyl_segments as f32);
        let angle2 = ((i + 1) as f32) * 2.0 * std::f32::consts::PI / (cyl_segments as f32);

        let x1 = radius * angle1.cos();
        let z1 = radius * angle1.sin();
        let x2 = radius * angle2.cos();
        let z2 = radius * angle2.sin();

        // v0: bottom at angle1, v1: bottom at angle2
        // v2: top at angle1,   v3: top at angle2
        vertices.push(Vec3::new(x1, 0.0, z1));
        vertices.push(Vec3::new(x2, 0.0, z2));
        vertices.push(Vec3::new(x1, cyl_height, z1));
        vertices.push(Vec3::new(x2, cyl_height, z2));

        let base = (vertices.len() - 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }

    // Cylinder caps: only add bottom cap (we'll attach hemisphere to top)
    let bottom_center = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, 0.0, 0.0));
    // top_center is not created (we don't want the top closed)

    for i in 0..cyl_segments {
        // idx is the start of the 4-vertex group for this segment
        let idx = bottom_center - (cyl_segments as u32 * 4) + (i as u32 * 4);
        // bottom cap triangle (center, next, current) - winding consistent with side faces
        indices.extend_from_slice(&[bottom_center, idx + 1, idx]);
        // NOTE: top cap intentionally omitted so hemisphere can attach
    }

    // Hemisphere (nose)
    let hemi_base_y = cyl_height;
    let hemi_start = vertices.len() as u32;

    // Create hemi_segments rings (exclude pole duplication). Each ring has hemi_segments vertices.
    // phi goes from (1/hemi_segments * PI/2) up to (hemi_segments/hemi_segments * PI/2) == PI/2
    for i in 0..hemi_segments {
        let phi = (i as f32 + 1.0) * std::f32::consts::PI / (2.0 * hemi_segments as f32); // avoid phi == 0
        let y = hemi_base_y + radius * phi.cos();
        let r = radius * phi.sin();
        for j in 0..hemi_segments {
            let theta = (j as f32) * 2.0 * std::f32::consts::PI / (hemi_segments as f32);
            let x = r * theta.cos();
            let z = r * theta.sin();
            vertices.push(Vec3::new(x, y, z));
        }
    }

    // Add single pole vertex (top of hemisphere)
    let pole_index = vertices.len() as u32;
    vertices.push(Vec3::new(0.0, hemi_base_y + radius, 0.0));

    // Connect cylinder top ring to first hemisphere ring
    // Cylinder top ring vertices are located every 4th vertex starting at cyl_side_start + 2
    for j in 0..cyl_segments {
        let cyl_idx_a = cyl_side_start + (j as u32 * 4) + 2; // top vertex for angle j
        let cyl_idx_b = cyl_side_start + (((j + 1) % cyl_segments) as u32 * 4) + 2; // next top vertex
        let hemi_idx_c = hemi_start + j as u32;
        let hemi_idx_d = hemi_start + ((j + 1) % hemi_segments) as u32;
        // two triangles to form quad between cylinder top and first hemi ring
        indices.extend_from_slice(&[cyl_idx_a, hemi_idx_c, cyl_idx_b]);
        indices.extend_from_slice(&[cyl_idx_b, hemi_idx_c, hemi_idx_d]);
    }

    // Build hemisphere faces between rings, and finally to pole
    for i in 0..hemi_segments {
        for j in 0..hemi_segments {
            let a = hemi_start + (i as u32 * hemi_segments as u32) + j as u32;
            let b = hemi_start + (i as u32 * hemi_segments as u32) + ((j + 1) % hemi_segments) as u32;
            if i == hemi_segments - 1 {
                // last ring -> triangle to pole
                indices.extend_from_slice(&[a, b, pole_index]);
            } else {
                let c = hemi_start + ((i + 1) as u32 * hemi_segments as u32) + j as u32;
                let d = hemi_start + ((i + 1) as u32 * hemi_segments as u32) + ((j + 1) % hemi_segments) as u32;
                indices.extend_from_slice(&[a, b, c, c, b, d]);
            }
        }
    }

    // Colours
    colours = vec![Vec4::new(0.8, 0.8, 0.9, 1.0); vertices.len()];

    // Compute smooth normals from geometry (per-vertex averaged face normals)
    normals = vec![Vec3::ZERO; vertices.len()];
    let mut tri = [0usize; 3];
    for chunk in indices.chunks(3) {
        if chunk.len() < 3 { continue; }
        tri[0] = chunk[0] as usize;
        tri[1] = chunk[1] as usize;
        tri[2] = chunk[2] as usize;
        let v0 = vertices[tri[0]];
        let v1 = vertices[tri[1]];
        let v2 = vertices[tri[2]];
        let face = (v1 - v0).cross(v2 - v0);
        // accumulate (non-normalized) face normal to each vertex
        normals[tri[0]] += face;
        normals[tri[1]] += face;
        normals[tri[2]] += face;
    }
    for n in &mut normals {
        if n.length_squared() > 0.0 {
            *n = n.normalize();
        } else {
            *n = Vec3::ZERO;
        }
    }

    (vertices, colours, normals, indices)
}


// --- UPLOAD MESH TO GPU ---
fn upload_mesh(vertices: &[Vec3], colours: &[Vec4], normals: &[Vec3], indices: &[u32]) -> (u32, i32) {
    let mut vao = 0;
    let mut vbo = [0; 3];
    let mut ebo = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(3, vbo.as_mut_ptr());
        gl::GenBuffers(1, &mut ebo);
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[0]);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * mem::size_of::<Vec3>()) as isize,
            vertices.as_ptr() as *const _,
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

// --- CREATE LIGHT CUBE (EMISSIVE) ---
fn create_light_cube() -> (u32, i32) {
    let size = 0.15;
    let vertices: Vec<Vec3> = vec![
        Vec3::new(-size, -size,  size),
        Vec3::new( size, -size,  size),
        Vec3::new( size,  size,  size),
        Vec3::new(-size,  size,  size),
        Vec3::new(-size, -size, -size),
        Vec3::new( size, -size, -size),
        Vec3::new( size,  size, -size),
        Vec3::new(-size,  size, -size),
    ];
    let colours: Vec<Vec4> = vec![Vec4::new(1.0, 1.0, 0.8, 1.0); vertices.len()];
    let normals: Vec<Vec3> = vec![Vec3::ZERO; vertices.len()];
    let indices: Vec<u32> = vec![
        0, 1, 2, 2, 3, 0,
        1, 5, 6, 6, 2, 1,
        7, 6, 5, 5, 4, 7,
        4, 0, 3, 3, 7, 4,
        4, 5, 1, 1, 0, 4,
        3, 2, 6, 6, 7, 3,
    ];
    upload_mesh(&vertices, &colours, &normals, &indices)
}

// --- SHADERS ---
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

// --- RENDER HELPER ---
fn render_mesh(
    vao: u32,
    count: i32,
    model: Mat4,
    program: u32,
    model_loc: i32,
    normal_loc: i32,
    emitmode_loc: i32,
    emissive: bool,
) {
    let normal_matrix = Mat3::from_mat4(model).inverse().transpose();
    unsafe {
        gl::UseProgram(program);
        gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.to_cols_array().as_ptr());
        gl::UniformMatrix3fv(normal_loc, 1, gl::FALSE, normal_matrix.to_cols_array().as_ptr());
        gl::Uniform1ui(emitmode_loc, if emissive { 1 } else { 0 });
        gl::BindVertexArray(vao);
        gl::DrawElements(gl::TRIANGLES, count, gl::UNSIGNED_INT, std::ptr::null());
    }
}

// --- CAMERA ---
struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    up: Vec3,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    const MOUSE_SENSITIVITY: f32 = 0.002;

    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 6));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlDebugContext(true));

    let (mut window, events) = glfw
        .create_window(1280, 720, "Turret - CS51012 Part 1", glfw::WindowMode::Windowed)
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

    // Load shaders
    let vertex_src = read_to_string("poslight.vert")?;
    let fragment_src = read_to_string("poslight.frag")?;
    let shader_program = create_shader_program(&vertex_src, &fragment_src)?;

    // Build turret parts
    let (base_vao, base_count) = {
        let (v, c, n, i) = create_box(2.0, 0.3, 2.0);
        upload_mesh(&v, &c, &n, &i)
    };
    let (cabin_vao, cabin_count) = {
        let (v, c, n, i) = create_box(1.0, 0.8, 1.0);
        upload_mesh(&v, &c, &n, &i)
    };
    let (boom_vao, boom_count) = {
        let (v, c, n, i) = create_box(0.2, 0.2, 2.0);
        upload_mesh(&v, &c, &n, &i)
    };
    let (arm_vao, arm_count) = {
        let (v, c, n, i) = create_box(0.15, 0.15, 1.5);
        upload_mesh(&v, &c, &n, &i)
    };
    let (light_vao, light_count) = create_light_cube();

    // Projectile
    let (proj_vao, proj_count) = {
        let (v, c, n, i) = create_projectile(0.05, 0.2, 8);
        upload_mesh(&v, &c, &n, &i)
    };

    // Camera
    let mut camera = Camera {
        position: Vec3::new(0.0, 1.0, 5.0),
        yaw: -std::f32::consts::FRAC_PI_2,
        pitch: 0.0,
        up: Vec3::new(0.0, 1.0, 0.0),
    };

    let projection = Mat4::perspective_rh(std::f32::consts::PI / 4.0, 1280.0 / 720.0, 0.1, 100.0);

    // Animation state
    let mut base_angle = 0.0_f32;
    let mut boom_angle = 0.4_f32;
    let mut arm_extension = 0.1_f32;
    let mut animation_speed = 1.0_f32;

    let mut keys_pressed: HashSet<Key> = HashSet::new();

    // Uniform locations
    let model_loc = unsafe { gl::GetUniformLocation(shader_program, b"model\0".as_ptr() as *const _) };
    let view_loc = unsafe { gl::GetUniformLocation(shader_program, b"view\0".as_ptr() as *const _) };
    let proj_loc = unsafe { gl::GetUniformLocation(shader_program, b"projection\0".as_ptr() as *const _) };
    let normal_loc = unsafe { gl::GetUniformLocation(shader_program, b"normalmatrix\0".as_ptr() as *const _) };
    let lightpos_loc = unsafe { gl::GetUniformLocation(shader_program, b"lightpos\0".as_ptr() as *const _) };
    let emitmode_loc = unsafe { gl::GetUniformLocation(shader_program, b"emitmode\0".as_ptr() as *const _) };
    let attenuationmode_loc = unsafe { gl::GetUniformLocation(shader_program, b"attenuationmode\0".as_ptr() as *const _) };

    // Projectile state
    #[derive(Default)]
    struct Projectile {
        active: bool,
        position: Vec3,
        velocity: Vec3,
        lifetime: f32,
    }
    let mut projectile = Projectile::default();
    const PROJECTILE_LIFETIME: f32 = 3.0;

    // Cache muzzle for accurate spawning
    let mut muzzle_world_pos = Vec3::ZERO;
    let mut muzzle_forward_dir = Vec3::new(0.0, 0.0, -1.0);

    let mut last_time = std::time::Instant::now();

    while !window.should_close() {
        let now = std::time::Instant::now();
        let delta = (now - last_time).as_secs_f32().min(0.1);
        last_time = now;

        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
                glfw::WindowEvent::Key(key, _, Action::Press, _) => { keys_pressed.insert(key); },
                glfw::WindowEvent::Key(key, _, Action::Release, _) => { keys_pressed.remove(&key); },
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
                    }
                    last_mouse_pos = Some((xpos, ypos));
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

        // Animation speed control
        if keys_pressed.contains(&Key::Equal) || keys_pressed.contains(&Key::KpAdd) {
            animation_speed = (animation_speed + 0.5).min(5.0);
        }
        if keys_pressed.contains(&Key::Minus) || keys_pressed.contains(&Key::KpSubtract) {
            animation_speed = (animation_speed - 0.5).max(0.1);
        }

        // Turret controls
        if keys_pressed.contains(&Key::Q) { base_angle += delta * animation_speed; }
        if keys_pressed.contains(&Key::E) { base_angle -= delta * animation_speed; }
        if keys_pressed.contains(&Key::Up) { boom_angle = (boom_angle + delta * animation_speed).min(1.2); }
        if keys_pressed.contains(&Key::Down) { boom_angle = (boom_angle - delta * animation_speed).max(-0.2); }
        if keys_pressed.contains(&Key::Left) { arm_extension = (arm_extension - delta * animation_speed).max(0.1); }
        if keys_pressed.contains(&Key::Right) { arm_extension = (arm_extension + delta * animation_speed).min(1.0); }

        // Fire projectile
        if keys_pressed.contains(&Key::F) && !projectile.active {
            projectile = Projectile {
                active: true,
                position: muzzle_world_pos,
                velocity: muzzle_forward_dir * 2.0 * animation_speed,
                lifetime: PROJECTILE_LIFETIME,
            };
        }

        // Update projectile
        if projectile.active {
            projectile.position += projectile.velocity * delta;
            projectile.lifetime -= delta;
            if projectile.lifetime <= 0.0 || projectile.position.y < -2.0 {
                projectile.active = false;
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
        let cam_speed = 0.05;

        if keys_pressed.contains(&Key::W) { camera.position += forward * cam_speed; }
        if keys_pressed.contains(&Key::S) { camera.position -= forward * cam_speed; }
        if keys_pressed.contains(&Key::A) { camera.position -= right * cam_speed; }
        if keys_pressed.contains(&Key::D) { camera.position += right * cam_speed; }
        if keys_pressed.contains(&Key::Space) { camera.position += world_up * cam_speed; }
        if keys_pressed.contains(&Key::LeftShift) { camera.position -= world_up * cam_speed; }

        // View matrix
        let target = camera.position + forward;
        let view = Mat4::look_at_rh(camera.position, target, camera.up);

        unsafe {
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }

        // --- RENDER TURRET HIERARCHY ---
        let mut ts = TransformStack::new();

        // Base
        ts.push(Mat4::from_translation(Vec3::new(0.0, -0.85, 0.0)));
        render_mesh(base_vao, base_count, ts.current(), shader_program, model_loc, normal_loc, emitmode_loc, false);

        // Cabin
        ts.push(Mat4::from_rotation_y(base_angle));
        ts.push(Mat4::from_translation(Vec3::new(0.0, 0.55, 0.0)));
        render_mesh(cabin_vao, cabin_count, ts.current(), shader_program, model_loc, normal_loc, emitmode_loc, false);

        // Boom
        ts.push(Mat4::from_translation(Vec3::new(0.0, 0.4, 0.0)));
        ts.push(Mat4::from_rotation_x(boom_angle));
        render_mesh(boom_vao, boom_count, ts.current(), shader_program, model_loc, normal_loc, emitmode_loc, false);

        // Arm
        ts.push(Mat4::from_translation(Vec3::new(0.0, 0.0, -1.75)));
        let original_half_depth = 1.5 / 2.0;
        let scale_z = arm_extension;
        let offset_z = original_half_depth - original_half_depth * scale_z;
        let transform = Mat4::from_translation(Vec3::new(0.0, 0.0, offset_z))
            * Mat4::from_scale(Vec3::new(1.0, 1.0, scale_z));
        ts.push(transform);
        render_mesh(arm_vao, arm_count, ts.current(), shader_program, model_loc, normal_loc, emitmode_loc, false);

        // Light (muzzle)
        ts.push(Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0 / scale_z)));
        let light_model = ts.current() * Mat4::from_translation(Vec3::new(0.0, 0.0, -0.1));
        let light_pos = light_model.transform_point3(Vec3::ZERO);
        let light_view_pos = (view * light_pos.extend(1.0)).truncate();

        // 💥 Cache muzzle for firing
        let muzzle_offset = Vec3::new(0.0, 0.0, -0.3);
        muzzle_world_pos = light_model.transform_point3(muzzle_offset);
        muzzle_forward_dir = (light_model * Vec4::new(0.0, 0.0, -1.0, 0.0)).truncate().normalize();

        unsafe {
            gl::UseProgram(shader_program);
            gl::UniformMatrix4fv(view_loc, 1, gl::FALSE, view.to_cols_array().as_ptr());
            gl::UniformMatrix4fv(proj_loc, 1, gl::FALSE, projection.to_cols_array().as_ptr());
            gl::Uniform4f(lightpos_loc, light_view_pos.x, light_view_pos.y, light_view_pos.z, 1.0);
            gl::Uniform1ui(attenuationmode_loc, 1);
        }
        render_mesh(light_vao, light_count, light_model, shader_program, model_loc, normal_loc, emitmode_loc, true);

        // Render projectile with correct orientation
        if projectile.active {
            let forward = projectile.velocity.normalize();
            let rot = if forward.dot(Vec3::Y).abs() > 0.999 {
                if forward.y > 0.0 {
                    Mat4::IDENTITY
                } else {
                    Mat4::from_rotation_x(std::f32::consts::PI)
                }
            } else {
                let axis = Vec3::Y.cross(forward).normalize();
                let angle = Vec3::Y.angle_between(forward);
                Mat4::from_axis_angle(axis, angle)
            };
            let proj_model = Mat4::from_translation(projectile.position) * rot;
            render_mesh(proj_vao, proj_count, proj_model, shader_program, model_loc, normal_loc, emitmode_loc, false);
        }

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteProgram(shader_program);
    }

    Ok(())
}