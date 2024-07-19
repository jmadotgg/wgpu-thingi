use crate::{Vertex, INDICES, VERTICES};

const CHUNK_SIZE: i32 = 64;

#[derive(Debug, Copy)]
enum Block {
    GRASS,
    AIR,
}

#[derive(Debug)]
pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug)]
pub struct Chunk {
    pos: Position,
    chunk_data: [Block; CHUNK_SIZE.pow(3) as usize],
}

impl Chunk {
    pub fn new(pos: Position) -> Self {
        let chunk_data = [Block::GRASS; CHUNK_SIZE.pow(3) as usize];
        Self { pos, chunk_data }
    }

    pub fn generate_chunk_data(&mut self) {
        todo!()
    }
}

pub struct Mesh {
    indices: Vec<u16>,
    vertices: Vec<Vertex>,
}

impl Mesh {
    fn fancy_algorithm_to_calculate_mesh_from_chunk() {
        todo!()
    }
}

impl From<&Chunk> for Mesh {
    fn from(value: &Chunk) -> Self {
        let Chunk { pos, chunk_data } = value;

        let color = [0.5, 0.5, 0.6];
        let indices = INDICES.to_vec();

        #[rustfmt::skip]
        let vertices = vec![
            // Front
            Vertex {
                position: [pos.x as f32 + -32.0, pos.y as f32 + 32.0, pos.z as f32 + 0.0],
                color: color,
            }, // A
            Vertex {
                position: [pos.x as f32 + -32.0, pos.y as f32 + -32.0, pos.z as f32 + 0.0],
                color: color,
            }, // B
            Vertex {
                position: [pos.x as f32 + 32.0, pos.y as f32 + 32.0, pos.z as f32 + 0.0],
                color: color,
            }, // C
            Vertex {
                position: [pos.x as f32 + 32.0, pos.y as f32 + -32.0, pos.z as f32 + 0.0],
                color: color,
            }, // D
            // Back
            Vertex {
                position: [pos.x as f32 + -32.0, pos.y as f32 + 32.0, pos.z as f32 + -64.0],
                color: color,
            }, // A
            Vertex {
                position: [pos.x as f32 + -32.0, pos.y as f32 + -32.0, pos.z as f32 + -64.0],
                color: color,
            }, // B
            Vertex {
                position: [pos.x as f32 + 32.0, pos.y as f32 + 32.0, pos.z as f32 + -64.0],
                color: color,
            }, // C
            Vertex {
                position: [pos.x as f32 + 32.0, pos.y as f32 + -32.0, pos.z as f32 + -64.0],
                color: color,
            }, // D
        ];

        Self { vertices, indices }
    }
}
