use criterion::{black_box, criterion_group, criterion_main, Criterion};
use leanrl_core::algorithms::utils;
use leanrl_core::obs::Obs;

fn bench_linear_transform(c: &mut Criterion) {
    let input = Obs::new([1.0f32, 2.0, 3.0, 4.0]);
    let weights = [[0.1f32; 4]; 2];
    let bias = [0.05f32, -0.05];
    c.bench_function("linear_transform_4x2", |b| {
        b.iter(|| {
            black_box(utils::linear_transform(
                black_box(&input),
                black_box(&weights),
                black_box(&bias),
            ))
        });
    });
}

fn bench_matrix_vector_backend(c: &mut Criterion) {
    let input = Obs::new([1.0f32, 2.0, 3.0, 4.0]);
    let weights = [[0.1f32; 4]; 2];
    let bias = [0.05f32, -0.05];
    c.bench_function("matrix_vector_mul_backend_4x2", |b| {
        b.iter(|| {
            black_box(utils::matrix_vector_mul_backend(
                black_box(&input),
                black_box(&weights),
                black_box(&bias),
            ))
        });
    });
}

criterion_group!(benches, bench_linear_transform, bench_matrix_vector_backend);
criterion_main!(benches);
