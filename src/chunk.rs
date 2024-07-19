use crate::{Vertex, INDICES};

const CHUNK_SIZE: usize = 64;
const CHUNK_DISPLACEMENT: f32 = (CHUNK_SIZE / 2) as f32;

#[derive(Debug, Copy, Clone, PartialEq)]
enum Block {
    GRASS = 1,
    AIR = 0,
}

#[derive(Debug)]
pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Position {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug)]
pub struct Chunk {
    pos: Position,
    chunk_data: [Block; CHUNK_SIZE.pow(3)],
}

impl Chunk {
    pub fn new(pos: Position) -> Self {
        let mut chunk_data = [Block::GRASS; CHUNK_SIZE.pow(3)];
        for i in 0..chunk_data.len() {
            chunk_data[i] = match rand::random::<f32>() {
                0.0..0.5 => Block::AIR,
                0.5..1.0 => Block::GRASS,
                _ => todo!(),
            }
        }
        Self { pos, chunk_data }
    }

    pub fn generate_chunk_data(&mut self) {
        todo!()
    }
}

pub struct Mesh {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
}

impl Mesh {
    fn fancy_algorithm_to_calculate_mesh_from_chunk() {
        todo!()
    }
}

impl From<&Chunk> for Mesh {
    fn from(value: &Chunk) -> Self {
        let Chunk { pos, chunk_data } = value;

        let color: [[f32; 3]; 8] = [
            [0.5, 0.0, 0.2],
            [0.0, 0.5, 0.2],
            [0.0, 0.0, 0.2],
            [0.5, 0.0, 0.2],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
        ];

        let mut vertices = Vec::new();

        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE { // order of x and y very important for vertex positions
                for z in 0..CHUNK_SIZE {
                    let block = chunk_data[y * z + x];

                    if block == Block::GRASS {
                        let (x, y, z) = (
                            pos.x as f32 + x as f32 - CHUNK_DISPLACEMENT,
                            pos.y as f32 + y as f32 - CHUNK_DISPLACEMENT,
                            pos.z as f32 + z as f32 - CHUNK_DISPLACEMENT,
                        );

                        #[rustfmt::skip]
                        vertices.push(vec![
                            Vertex { position: [x + -0.5, y + 0.5,  z + 0.0], color: color[0] }, // A
                            Vertex { position: [x + -0.5, y + -0.5, z + 0.0], color: color[1] }, // B
                            Vertex { position: [x + 0.5,  y + 0.5,  z + 0.0], color: color[2] }, // C
                            Vertex { position: [x + 0.5,  y + -0.5, z + 0.0], color: color[3] }, // D
                            Vertex { position: [x + -0.5, y + 0.5,  z + -1.0], color: color[4] }, // A
                            Vertex { position: [x + -0.5, y + -0.5, z + -1.0], color: color[5] }, // B
                            Vertex { position: [x + 0.5,  y + 0.5,  z + -1.0], color: color[6] }, // C
                            Vertex { position: [x + 0.5,  y + -0.5, z + -1.0], color: color[7] }, // D
                        ]);
                        //vertices.push(Vertex {
                        //    position: [x, y, z],
                        //    color,
                        //})
                    }
                }
            }
        }

        let indices = (0..vertices.len())
            .flat_map(|i| {
                INDICES
                    .to_vec()
                    .iter()
                    .map(|e| i as u32 * INDICES.len() as u32 + *e as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let vertices = vertices.into_iter().flatten().collect::<Vec<_>>();
        dbg!((0..108).map(|i| vertices.get(i).unwrap().position).collect::<Vec<_>>());

        //#[rustfmt::skip]
        //let vertices = vec![
        //    // Front
        //    Vertex {
        //        position: [pos.x as f32 + -32.0, pos.y as f32 + 32.0, pos.z as f32 + 0.0],
        //        color: color,
        //    }, // A
        //    Vertex {
        //        position: [pos.x as f32 + -32.0, pos.y as f32 + -32.0, pos.z as f32 + 0.0],
        //        color: color,
        //    }, // B
        //    Vertex {
        //        position: [pos.x as f32 + 32.0, pos.y as f32 + 32.0, pos.z as f32 + 0.0],
        //        color: color,
        //    }, // C
        //    Vertex {
        //        position: [pos.x as f32 + 32.0, pos.y as f32 + -32.0, pos.z as f32 + 0.0],
        //        color: color,
        //    }, // D
        //    // Back
        //    Vertex {
        //        position: [pos.x as f32 + -32.0, pos.y as f32 + 32.0, pos.z as f32 + -64.0],
        //        color: color,
        //    }, // A
        //    Vertex {
        //        position: [pos.x as f32 + -32.0, pos.y as f32 + -32.0, pos.z as f32 + -64.0],
        //        color: color,
        //    }, // B
        //    Vertex {
        //        position: [pos.x as f32 + 32.0, pos.y as f32 + 32.0, pos.z as f32 + -64.0],
        //        color: color,
        //    }, // C
        //    Vertex {
        //        position: [pos.x as f32 + 32.0, pos.y as f32 + -32.0, pos.z as f32 + -64.0],
        //        color: color,
        //    }, // D
        //];

        Self { vertices, indices }
    }
}
