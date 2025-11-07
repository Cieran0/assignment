mod camera;
mod mesh;
mod transform;
mod projectile;
mod shaders;

use std::{collections::HashSet, error::Error, fs::read_to_string, time::Instant};

use glam::{Mat3, Mat4, Vec3, Vec4};
use glfw::{Action, Context, Key, OpenGlProfileHint};

use crate::{
    camera::Camera,
    mesh::{create_box, create_projectile, create_light_cube, upload_mesh},
    transform::TransformStack,
    projectile::Projectile,
    shaders::{compile_shader, create_shader_program},
};

const PROJECTILE_LIFETIME: f32 = 3.0;
const MOUSE_SENSITIVITY: f32 = 0.002;

fn main() -> Result<(), Box<dyn Error>> {
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;

    let mut glfw = glfw::init(glfw::fail_on_errors)?;
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 6));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlDebugContext(true));
    glfw.window_hint(glfw::WindowHint::Floating(true));

    let width = 1920;
    let height = 1080;

    let (mut window, events) = glfw
        .create_window(width, height, "GL Test", glfw::WindowMode::Windowed)
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

    // Shaders
    let vertex_src = read_to_string("poslight.vert")?;
    let fragment_src = read_to_string("poslight.frag")?;
    let shader_program = create_shader_program(&vertex_src, &fragment_src)?;

    // Meshes
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
    let (light_vao, light_count) = {
        let (v, c, n, i) = create_light_cube();
        upload_mesh(&v, &c, &n, &i)
    };
    let (proj_vao, proj_count) = {
        let (v, c, n, i) = create_projectile(0.05, 0.2, 8);
        upload_mesh(&v, &c, &n, &i)
    };

    // Camera
    let mut camera = Camera::new(Vec3::new(0.0, 1.0, 5.0));

    let projection = Mat4::perspective_rh(std::f32::consts::PI / 4.0, width as f32 / height as f32, 0.1, 100.0);

    // Animation state
    let mut base_angle = 0.0_f32;
    let mut boom_angle = 0.4_f32;
    let mut arm_extension = 0.1_f32;
    let mut animation_speed = 1.0_f32;

    let mut keys_pressed: HashSet<Key> = HashSet::new();
    let mut projectile = Projectile::default();

    let mut muzzle_world_pos = Vec3::ZERO;
    let mut muzzle_forward_dir = Vec3::new(0.0, 0.0, -1.0);

    // Uniform locations
    let model_loc = unsafe { gl::GetUniformLocation(shader_program, b"model\0".as_ptr() as *const _) };
    let view_loc = unsafe { gl::GetUniformLocation(shader_program, b"view\0".as_ptr() as *const _) };
    let proj_loc = unsafe { gl::GetUniformLocation(shader_program, b"projection\0".as_ptr() as *const _) };
    let normal_loc = unsafe { gl::GetUniformLocation(shader_program, b"normalmatrix\0".as_ptr() as *const _) };
    let lightpos_loc = unsafe { gl::GetUniformLocation(shader_program, b"lightpos\0".as_ptr() as *const _) };
    let emitmode_loc = unsafe { gl::GetUniformLocation(shader_program, b"emitmode\0".as_ptr() as *const _) };
    let attenuationmode_loc = unsafe { gl::GetUniformLocation(shader_program, b"attenuationmode\0".as_ptr() as *const _) };

    // Rendering helper closure
    let render_mesh = |vao: u32, count: i32, model: Mat4, emissive: bool| {
        let normal_matrix = Mat3::from_mat4(model).inverse().transpose();
        unsafe {
            gl::UseProgram(shader_program);
            gl::UniformMatrix4fv(model_loc, 1, gl::FALSE, model.to_cols_array().as_ptr());
            gl::UniformMatrix3fv(normal_loc, 1, gl::FALSE, normal_matrix.to_cols_array().as_ptr());
            gl::Uniform1ui(emitmode_loc, if emissive { 1 } else { 0 });
            gl::BindVertexArray(vao);
            gl::DrawElements(gl::TRIANGLES, count, gl::UNSIGNED_INT, std::ptr::null());
        }
    };

    let mut last_time = Instant::now();

    while !window.should_close() {
        let now = Instant::now();
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
                            camera.clamp_pitch();
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

        // Animation speed
        if keys_pressed.contains(&Key::Equal) || keys_pressed.contains(&Key::KpAdd) {
            animation_speed = (animation_speed + 0.5).min(5.0);
        }
        if keys_pressed.contains(&Key::Minus) || keys_pressed.contains(&Key::KpSubtract) {
            animation_speed = (animation_speed - 0.5).max(0.1);
        }

        // Turret controls
        if keys_pressed.contains(&Key::Q) { base_angle += delta * animation_speed; }
        if keys_pressed.contains(&Key::E) { base_angle -= delta * animation_speed; }
        if keys_pressed.contains(&Key::Up) { boom_angle = boom_angle.min(1.2) + delta * animation_speed; }
        if keys_pressed.contains(&Key::Down) { boom_angle = boom_angle.max(-0.2) - delta * animation_speed; }
        if keys_pressed.contains(&Key::Left) { arm_extension = arm_extension.max(0.1) - delta * animation_speed; }
        if keys_pressed.contains(&Key::Right) { arm_extension = arm_extension.min(1.0) + delta * animation_speed; }

        // Fire
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
        let forward = camera.forward();
        let right = forward.cross(camera.up).normalize();
        let world_up = Vec3::Y;
        let cam_speed = 0.05;

        if keys_pressed.contains(&Key::W) { camera.position += forward * cam_speed; }
        if keys_pressed.contains(&Key::S) { camera.position -= forward * cam_speed; }
        if keys_pressed.contains(&Key::A) { camera.position -= right * cam_speed; }
        if keys_pressed.contains(&Key::D) { camera.position += right * cam_speed; }
        if keys_pressed.contains(&Key::Space) { camera.position += world_up * cam_speed; }
        if keys_pressed.contains(&Key::LeftShift) { camera.position -= world_up * cam_speed; }

        let view = camera.view_matrix();

        unsafe {
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::UseProgram(shader_program);
            gl::UniformMatrix4fv(view_loc, 1, gl::FALSE, view.to_cols_array().as_ptr());
            gl::UniformMatrix4fv(proj_loc, 1, gl::FALSE, projection.to_cols_array().as_ptr());
            gl::Uniform1ui(attenuationmode_loc, 1);
        }

        // Render turret
        let mut ts = TransformStack::new();

        ts.push(Mat4::from_translation(Vec3::new(0.0, -0.85, 0.0)));
        render_mesh(base_vao, base_count, ts.current(), false);

        ts.push(Mat4::from_rotation_y(base_angle));
        ts.push(Mat4::from_translation(Vec3::new(0.0, 0.55, 0.0)));
        render_mesh(cabin_vao, cabin_count, ts.current(), false);

        ts.push(Mat4::from_translation(Vec3::new(0.0, 0.4, 0.0)));
        ts.push(Mat4::from_rotation_x(boom_angle));
        render_mesh(boom_vao, boom_count, ts.current(), false);

        ts.push(Mat4::from_translation(Vec3::new(0.0, 0.0, -1.75)));
        let original_half_depth = 1.5 / 2.0;
        let scale_z = arm_extension;
        let offset_z = original_half_depth - original_half_depth * scale_z;
        let transform = Mat4::from_translation(Vec3::new(0.0, 0.0, offset_z))
            * Mat4::from_scale(Vec3::new(1.0, 1.0, scale_z));
        ts.push(transform);
        render_mesh(arm_vao, arm_count, ts.current(), false);

        // Light (muzzle)
        ts.push(Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0 / scale_z)));
        let light_model = ts.current() * Mat4::from_translation(Vec3::new(0.0, 0.0, -0.1));
        let light_pos = light_model.transform_point3(Vec3::ZERO);
        let light_view_pos = (view * light_pos.extend(1.0)).truncate();

        muzzle_world_pos = light_model.transform_point3(Vec3::new(0.0, 0.0, -0.3));
        muzzle_forward_dir = (light_model * Vec4::new(0.0, 0.0, -1.0, 0.0)).truncate().normalize();

        unsafe {
            gl::Uniform4f(lightpos_loc, light_view_pos.x, light_view_pos.y, light_view_pos.z, 1.0);
        }
        render_mesh(light_vao, light_count, light_model, true);

        // Render projectile
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
            render_mesh(proj_vao, proj_count, proj_model, false);
        }

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteProgram(shader_program);
    }

    Ok(())
}