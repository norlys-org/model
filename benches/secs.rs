#![feature(test)]
extern crate test;
use norlys::secs::{secs_interpolate, Observation};
use test::Bencher;

fn make_input() -> Vec<Observation> {
    vec![Observation {
        latitude: 69.0,
        longitude: 19.0,
        i: -100.0,
        j: -100.0,
        altitude: 0.0,
    }]
}

#[bench]
fn bench_secs_interpolate(b: &mut Bencher) {
    let obs = make_input();
    b.iter(|| {
        let out = secs_interpolate(
            obs.clone(),
            45.0f32..85.0f32,
            5,
            -180.0f32..179.0f32,
            5,
            110e3,
            0.0f32,
        );
        test::black_box(out);
    })
}
