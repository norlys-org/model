/// Return evenly spaced numbers over a specified interval.
/// `start` needs to be strictly inferior to `end`
pub fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(num);

    if end <= start {
        panic!("End value needs to be strictly superior to start value.");
    }

    let step = (end - start) / ((num - 1) as f32);
    for i in 0..num {
        result.push(start + (i as f32) * step);
    }
    
    result
}
