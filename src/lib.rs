mod camera;
mod chunk;
mod texture;
use std::collections::HashMap;

use cgmath::prelude::*;
use cgmath::SquareMatrix;
use chunk::Chunk;
use chunk::ChunkWatcher;
use chunk::Mesh;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(window).await;
    let mut last_render_time = instant::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if state.mouse_pressed {
                    state.camera_controller.process_mouse(delta.0, delta.1)
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() && !state.input(event) => {
                if !state.input(event) {
                    match event {
                        WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size)
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                let now = instant::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}

// linear interpolation
fn lerp(t: f32, a: f32, b: f32) -> f32 {
    a + t * (b - a)
}

//#[derive(Debug)]
//struct Chunk {
//    pub instance_data: Vec<InstanceRaw>,
//}
//
//impl Chunk {
//    pub fn new<T>(offset: T) -> Self
//    where
//        T: Into<cgmath::Point3<f32>>,
//    {
//        let offset: cgmath::Point3<f32> = offset.into();
//
//        // https://jaysmito101.hashnode.dev/perlins-noise-algorithm
//        let mut gradients: [cgmath::Vector2<f32>; NUM_INSTANCES_PER_ROW as usize * 2 + 2] = (0
//            ..NUM_INSTANCES_PER_ROW as usize * 2 + 2)
//            .map(|_| cgmath::Vector2::new(rand::random::<f32>(), rand::random::<f32>()))
//            .collect::<Vec<_>>()
//            .try_into()
//            .unwrap();
//
//        let mut permutation: [usize; NUM_INSTANCES_PER_ROW as usize * 2 + 2] = (0
//            ..NUM_INSTANCES_PER_ROW as usize * 2 + 2)
//            .collect::<Vec<_>>()
//            .try_into()
//            .unwrap();
//
//        for (i, mut j) in (0..NUM_INSTANCES_PER_ROW as usize).enumerate() {
//            let k = permutation[i];
//            j = rand::random::<usize>() % NUM_INSTANCES_PER_ROW as usize;
//            permutation[i] = permutation[j];
//            permutation[j] = k;
//        }
//
//        for i in 0..NUM_INSTANCES_PER_ROW as usize + 2 {
//            permutation[NUM_INSTANCES_PER_ROW as usize + i] = permutation[i];
//            gradients[NUM_INSTANCES_PER_ROW as usize + i] = gradients[i];
//        }
//
//        let noise_calculation = (0..NUM_INSTANCES_PER_ROW)
//            .map(|z| {
//                (0..NUM_INSTANCES_PER_ROW)
//                    .map(move |x| {
//                        let rx0 = gradients[x as usize].x;
//                        let rx1 = rx0 - 1f32;
//                        let ry0 = gradients[z as usize].y;
//                        let ry1 = ry0 - 1f32;
//
//                        let bx0 = x as usize % 255;
//                        let by0 = z as usize % 255;
//                        let bx1 = (bx0 + 1) % 255;
//                        let by1 = (by0 + 1) % 255;
//
//                        let i = permutation[bx0];
//                        let j = permutation[bx1];
//
//                        let b00 = permutation[i + by0];
//                        let b10 = permutation[j + by1];
//                        let b01 = permutation[i + by0];
//                        let b11 = permutation[j + by1];
//
//                        let u =
//                            cgmath::Vector2::dot(gradients[b00], cgmath::Vector2::new(rx0, ry0));
//                        let v =
//                            cgmath::Vector2::dot(gradients[b10], cgmath::Vector2::new(rx1, ry0));
//                        let a = lerp(rx0, u, v);
//                        let u =
//                            cgmath::Vector2::dot(gradients[b01], cgmath::Vector2::new(rx0, ry1));
//                        let v =
//                            cgmath::Vector2::dot(gradients[b11], cgmath::Vector2::new(rx1, ry1));
//                        let b = lerp(rx0, u, v);
//                        let res = lerp(ry0, a, b) * 5.0;
//                        let res = res * 3.0;
//
//                        (-10..res as i32).map(|i| i as f32).collect::<Vec<_>>()
//                        //vec![-1.0f32, 0.0, 1.0, 2.0, 3.0, 4.0]
//                    })
//                    .collect::<Vec<_>>()
//            })
//            .collect::<Vec<_>>();
//
//        #[rustfmt::skip]
//        let instances_vertices = noise_calculation.into_iter().enumerate().flat_map(|(z, row)| {
//            row.into_iter().enumerate().flat_map(move |(x, noise)| {
//                noise.into_iter().map(move |y| {
//                    let position = cgmath::Vector3 {
//                        x: offset.x + x as f32,
//                        y: offset.y + y.floor(), //(f32::sin(x as f32) + f32::sin(z as f32)).round(),
//                        z: offset.z + z as f32,
//                    } - INSTANCE_DISPLACEMENT;
//                    //println!("z{} x{}", position.z, position.x);
//
//                    let rotation = if position.is_zero() {
//                        // this is needed so an object at (0, 0, 0) won't get scaled to zero
//                        // as Quaternions can effect scale if they're not created correctly
//                        cgmath::Quaternion::from_axis_angle(
//                            cgmath::Vector3::unit_z(),
//                            cgmath::Deg(0.0),
//                        )
//                    } else {
//                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(0.0))
//                    };
//
//                //                    let color = if position.y > 0.0 {
////                        [0.2, 0.3, 0.2]
////                    } else {
////                        [0.05, 0.04, 0.05]
////                    };
////
//                    //#[rustfmt::skip]
//                    //let vertices = vec![
//                    //    // Front
//                    //    Vertex { position: [position.x + -0.5, position.y +  0.5,  position.z + 0.0], color: color }, // A
//                    //    Vertex { position: [position.x + -0.5, position.y + -0.5, position.z + 0.0], color: color }, // B
//                    //    Vertex { position: [position.x +  0.5,  position.y +  0.5,  position.z + 0.0], color: color }, // C
//                    //    Vertex { position: [position.x +  0.5,  position.y + -0.5, position.z + 0.0], color: color }, // D
//                    //    // Back
//                    //    Vertex { position: [position.x + -0.5, position.y +  0.5,  position.z +  -1.0], color: color }, // A
//                    //    Vertex { position: [position.x + -0.5, position.y + -0.5, position.z +   -1.0], color: color }, // B
//                    //    Vertex { position: [position.x +  0.5,  position.y +  0.5,  position.z + -1.0], color: color }, // C
//                    //    Vertex { position: [position.x +  0.5,  position.y + -0.5, position.z +  -1.0], color: color }, // D
//                    //];
//
//
//                    Instance { position, rotation }
//                }).collect::<Vec<_>>()
//            }).collect::<Vec<_>>()
//        }).collect::<Vec<_>>();
//
//        let instance_data = instances_vertices
//            .iter()
//            .map(Instance::to_raw)
//            .collect::<Vec<_>>();
//
//        Self { instance_data }
//    }
//}

type ChunkId = String;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    chunk_buffers: Vec<(wgpu::Buffer, wgpu::Buffer, u32)>,
    num_indices: u32,
    camera: camera::Camera,
    projection: camera::Projection,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_uniform: CameraUniform,
    camera_controller: camera::CameraController,
    mouse_pressed: bool,
    chunk_watcher: ChunkWatcher,
    //instances: Vec<Instance>,
    //num_instances: u32,
    // instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    //chunks: HashMap<ChunkId, Chunk>,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let camera = camera::Camera::new((0.0, 2.0, 0.0), cgmath::Deg(-90.0), cgmath::Deg(0.0));
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(70.0), 0.1, 100.0);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);
        let camera_controller = camera::CameraController::new(10.0, 0.8);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc() /*, InstanceRaw::desc()*/],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let mut chunk_watcher = ChunkWatcher::new();
        let meshes = chunk_watcher.get_required_chunk_mesh_data(&chunk::Position::new(0, 0, 0));

        let mut chunk_buffers = Vec::new();
        for (id, mesh) in meshes.iter().enumerate() {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Vertex Buffer {id}")),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Index Buffer {id}")),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            chunk_buffers.push((vertex_buffer, index_buffer, mesh.indices.len() as u32));
        }

        let chunk = Chunk::new(&chunk::Position::new(0, 0, 0));
        let mesh = Mesh::from(&chunk);

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = mesh.indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            chunk_watcher,
            chunk_buffers,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            camera,
            projection,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            mouse_pressed: false,
            //instances,
            depth_texture,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }

            _ => false,
        }
    }

    fn update(&mut self, dt: instant::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);

        //println!("z{}x{}", self.camera.position.z as i32, self.camera.position.x as i32);

        let x = self.camera.position.x as i32 / (NUM_INSTANCES_PER_ROW as i32 / 2);
        let z = self.camera.position.z as i32 / (NUM_INSTANCES_PER_ROW as i32 / 2);
        let potential_new_chunk = format!("{x}{z}");

        // TODO: uff ugly as hell
        //if potential_new_chunk != self.current_chunk {
        //    if let Some(chunk) = self.chunks.get(&format!("{x}{z}")) {
        //        self.queue.write_buffer(
        //            &self.instance_buffer,
        //            0,
        //            bytemuck::cast_slice(&chunk.instance_data.as_slice()),
        //        );
        //    } else {
        //        let new_chunk = Chunk::new((
        //            (x + x * (NUM_INSTANCES_PER_ROW as i32 / 2)) as f32,
        //            0.0,
        //            (z + z * (NUM_INSTANCES_PER_ROW as i32 / 2)) as f32,
        //        ));
        //        self.queue.write_buffer(
        //            &self.instance_buffer,
        //            0,
        //            bytemuck::cast_slice(&new_chunk.instance_data.as_slice()),
        //        );
        //        self.chunks.insert(potential_new_chunk.clone(), new_chunk);
        //    };

        //    println!(
        //        "New Chunk {} | Previous chunk {}",
        //        potential_new_chunk, self.current_chunk,
        //    );
        //    self.current_chunk = potential_new_chunk;
        //}

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                // This is what @location(0) in the fragment shader targets
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.6,
                            g: 0.6,
                            b: 0.6,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            //render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            //render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            //render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

            //for i in 0..NUM_INSTANCES_PER_ROW {
            //    for j in 0..NUM_INSTANCES_PER_ROW {
            //        //self.queue.write_buffer(
            //        //    &self.instance_buffer,
            //        //    0,
            //        //    bytemuck::cast_slice(&new_chunk.instance_data.as_slice()),
            //        //);
            //        render_pass.set_vertex_buffer(
            //            0,
            //            self.vertex_buffers
            //                .get((i + j) as usize)
            //                .expect("Whoops vertex buffer shenanigans")
            //                .slice(..),
            //        );
            //        render_pass.
            //        render_pass.draw_indexed(0..self.num_indices, 0, (i + j)..(i + j + 1));

            //
            //    }
            //}
            //
            //
            for chunk_buffer in self.chunk_buffers.iter() {
                let (vertex_buffer, index_buffer, num_indices) = chunk_buffer;
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..*num_indices, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

//#[rustfmt::skip]
//const VERTICES: &[Vertex] = &[
//    // Front
//    Vertex { position: [-0.5, 0.5, 0.0], color:  [0.5, 0.0, 0.2] }, // A
//    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 0.5, 0.2] }, // B
//    Vertex { position: [0.5, 0.5, 0.0], color:   [0.0, 0.0, 0.2] }, // C
//    Vertex { position: [0.5, -0.5, 0.0], color:  [0.5, 0.0, 0.2] }, // D
//    // Back
//    Vertex { position: [-0.5, 0.5, -1.0], color: [0.5, 0.0, 0.5] }, // A
//    Vertex { position: [-0.5, -0.5, -1.0], color:[0.0, 0.5, 0.5] }, // B
//    Vertex { position: [0.5, 0.5, -1.0], color:  [0.0, 0.0, 0.5] }, // C
//    Vertex { position: [0.5, -0.5, -1.0], color: [0.5, 0.0, 0.5] }, // D
//];

#[rustfmt::skip]
const INDICES: &[u16] = &[
    // front
    0, 1, 2,
    1, 3, 2,
    // back
    6, 5, 4,
    5, 6, 7,
    // left
    0, 4, 1,
    1, 4, 5,
    // right
    6, 2, 3,
    3, 7, 6,
    // top
    0, 2, 4,
    6, 4, 2,
    // bottom
    1, 5, 3,
    5, 7, 3
];

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraUniform {
    // 16 bytes because of uniform 16 byte spacing requirement
    view_pos: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_pos: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_pos = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 64;
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
    0.0,
    NUM_INSTANCES_PER_ROW as f32 * 0.5,
);
