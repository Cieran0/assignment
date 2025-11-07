use std::mem;
use glam::{Mat4, Vec3, Vec4};

// --- CREATE A BOX (indexed) ---
pub fn create_box(width: f32, height: f32, depth: f32) -> (Vec<Vec3>, Vec<Vec4>, Vec<Vec3>, Vec<u32>) {
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
pub fn create_projectile(radius: f32, cyl_height: f32, hemi_segments: usize) -> (Vec<Vec3>, Vec<Vec4>, Vec<Vec3>, Vec<u32>) {
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

// --- CREATE LIGHT CUBE (EMISSIVE) ---
pub fn create_light_cube() -> (Vec<Vec3>, Vec<Vec4>, Vec<Vec3>, Vec<u32>) {
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
    (vertices, colours, normals, indices)
}

// --- UPLOAD MESH TO GPU ---
pub fn upload_mesh(vertices: &[Vec3], colours: &[Vec4], normals: &[Vec3], indices: &[u32]) -> (u32, i32) {
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