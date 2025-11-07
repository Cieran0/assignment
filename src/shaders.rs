use std::ffi::{CStr, CString};

pub fn compile_shader(source: &str, shader_type: u32) -> Result<u32, String> {
    let shader = unsafe { gl::CreateShader(shader_type) };
    let c_source = CString::new(source).expect("Shader source must be valid UTF-8");
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
            let error = CStr::from_ptr(error.as_ptr()).to_str().unwrap().to_string();
            return Err(format!("Shader compilation failed: {}", error));
        }
    }
    Ok(shader)
}

pub fn create_shader_program(vertex_src: &str, fragment_src: &str) -> Result<u32, String> {
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
            let error = CStr::from_ptr(error.as_ptr()).to_str().unwrap().to_string();
            return Err(format!("Shader linking failed: {}", error));
        }
        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
    }
    Ok(program)
}