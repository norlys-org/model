mod grid;
mod matrix;
mod secs;
mod sphere;

fn main() {
    println!(
        "{:?}",
        grid::geographical_point(0.0..10.0, 11, 0.0..10.0, 11, 110000f32)
    );
}
