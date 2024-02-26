pub fn transpose<const N: usize>(
    s: [[f32; N]; N],
) -> [[f32; N]; N] {
    let mut new_s: [[f32; N]; N] = [[0.; N]; N];
    for i in 0..N {
        for j in 0..N {
            new_s[i][j] = s[j][i];
        }
    }
    new_s
}

pub fn normalise<const N: usize>(
    am_vec: [f32; N],
    seed: [f32; N],
) -> [f32; N] {
    let sum: f32 = am_vec.iter().sum();
    if sum == 0. {
        return seed;
    }
    am_vec.map(|x| x / sum)
}

pub fn normalise_sqrt<const N: usize>(
    vector: [f32; N],
) -> [f32; N] {
    let sum: f32 = vector.iter().map(|x| x.powf(2.)).sum();
    if sum == 0. {
        return [0.; N];
    }
    vector.map(|x| x / sum.sqrt())
}

pub fn vec_scalar_mul<const N: usize>(s: [f32; N], y: f32) -> [f32; N] {
    s.map(|x| x * y)
}

pub fn vec_add<const N: usize>(s: [f32; N], y: [f32; N]) -> [f32; N] {
    let mut out: [f32; N] = [0.; N];
    for i in 0..N {
        out[i] = s[i] + y[i];
    }
    out
}
