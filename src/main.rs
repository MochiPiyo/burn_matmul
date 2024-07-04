
fn main() {
    let nums = [1, 2, 4, 8]; 
    for num in nums.iter() {
        let num = num * 1024;
        let time = std::time::Instant::now();
        let shape = burn::tensor::Shape::new([num, num]);
        let device = burn::backend::wgpu::WgpuDevice::default();
        
        type B = burn::backend::Wgpu;
        let a: burn::prelude::Tensor<B, 2> = burn::tensor::Tensor::ones(shape.clone(), &device);
        let b: burn::prelude::Tensor<B, 2> = burn::tensor::Tensor::ones(shape, &device);

        let c = a.matmul(b);

        println!("time = {:?}, {}", time.elapsed(), c.slice([0..1, 0..1]));
    }
    
}
