//! NumPy SeedSequence + PCG64 XSL-RR and Generator integer sampling.
//! Algorithm attribution/license: ../NUMPY-RNG-LICENSE.txt.
use std::collections::HashSet;
const MULT: u128 = 0x2360ed051fc65da44385df649fccf645;
#[derive(Clone)]
pub struct Rng {
    state: u128,
    inc: u128,
    buffer: Option<u32>,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        fn hash(mut x: u32, h: &mut u32) -> u32 {
            x ^= *h;
            *h = h.wrapping_mul(0x931e8875);
            x = x.wrapping_mul(*h);
            x ^ (x >> 16)
        }
        fn mix(x: u32, y: u32) -> u32 {
            let v = 0xca01f9ddu32
                .wrapping_mul(x)
                .wrapping_sub(0x4973f715u32.wrapping_mul(y));
            v ^ (v >> 16)
        }
        let entropy = if seed >> 32 == 0 {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        let mut pool = [0u32; 4];
        let mut h = 0x43b0d7e5;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hash(*entropy.get(i).unwrap_or(&0), &mut h);
        }
        for src in 0..4 {
            for dst in 0..4 {
                if src != dst {
                    pool[dst] = mix(pool[dst], hash(pool[src], &mut h));
                }
            }
        }
        let mut words = [0u32; 8];
        h = 0x8b51f9dd;
        for i in 0..8 {
            let mut v = pool[i % 4] ^ h;
            h = h.wrapping_mul(0x58f38ded);
            v = v.wrapping_mul(h);
            words[i] = v ^ (v >> 16);
        }
        let w: Vec<_> = (0..4)
            .map(|i| words[2 * i] as u64 | ((words[2 * i + 1] as u64) << 32))
            .collect();
        let init = ((w[0] as u128) << 64) | w[1] as u128;
        let stream = ((w[2] as u128) << 64) | w[3] as u128;
        let mut r = Self {
            state: 0,
            inc: (stream << 1) | 1,
            buffer: None,
        };
        r.step();
        r.state = r.state.wrapping_add(init);
        r.step();
        r
    }
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
    }
    pub fn raw(&mut self) -> u64 {
        self.step();
        ((self.state >> 64) as u64 ^ self.state as u64).rotate_right((self.state >> 122) as u32)
    }
    pub fn u32(&mut self) -> u32 {
        if let Some(v) = self.buffer.take() {
            v
        } else {
            let v = self.raw();
            self.buffer = Some((v >> 32) as u32);
            v as u32
        }
    }
    pub fn bounded(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        assert!(max < u32::MAX as usize);
        let n = max as u32 + 1;
        let threshold = (u32::MAX - max as u32) % n;
        loop {
            let m = self.u32() as u64 * n as u64;
            if m as u32 >= threshold {
                return (m >> 32) as usize;
            }
        }
    }
    fn interval(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        let mut mask = max as u32;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.u32() & mask;
            if v <= max as u32 {
                return v as usize;
            }
        }
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.interval(i);
            v.swap(i, j);
        }
    }
    fn shuffle_int<T>(&mut self, v: &mut [T], first: usize) {
        for i in (first..v.len()).rev() {
            let j = self.bounded(i);
            v.swap(i, j);
        }
    }
    pub fn choice(&mut self, n: usize, size: usize, replace: bool) -> Vec<usize> {
        assert!(n > 0 && (replace || size <= n));
        if replace {
            return (0..size).map(|_| self.bounded(n - 1)).collect();
        }
        if n > 10000 && size > n / 50 {
            let mut a: Vec<_> = (0..n).collect();
            self.shuffle_int(&mut a, (n - size).max(1));
            a[n - size..].to_vec()
        } else {
            let mut used = HashSet::new();
            let mut out = Vec::with_capacity(size);
            for j in n - size..n {
                let v = self.bounded(j);
                let v = if used.contains(&v) { j } else { v };
                used.insert(v);
                out.push(v);
            }
            self.shuffle_int(&mut out, 1);
            out
        }
    }
}
pub fn resample(y: &[u8], scenario: &str, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let (mut pos, mut neg): (Vec<usize>, Vec<usize>) = (
        (0..y.len()).filter(|&i| y[i] == 1).collect(),
        (0..y.len()).filter(|&i| y[i] == 0).collect(),
    );
    assert!(!pos.is_empty() && !neg.is_empty());
    let mut idx = match scenario {
        "none" => (0..y.len()).collect(),
        "upsampling" => {
            let target = pos.len().max(neg.len());
            if pos.len() < target {
                let extra = rng.choice(pos.len(), target - pos.len(), true);
                let values: Vec<_> = extra.into_iter().map(|i| pos[i]).collect();
                pos.extend(values);
            }
            if neg.len() < target {
                let extra = rng.choice(neg.len(), target - neg.len(), true);
                let values: Vec<_> = extra.into_iter().map(|i| neg[i]).collect();
                neg.extend(values);
            }
            pos.extend(neg);
            pos
        }
        "downsampling" => {
            let target = pos.len().min(neg.len());
            let mut a: Vec<_> = rng
                .choice(pos.len(), target, false)
                .iter()
                .map(|&i| pos[i])
                .collect();
            a.extend(rng.choice(neg.len(), target, false).iter().map(|&i| neg[i]));
            a
        }
        _ => panic!("invalid scenario"),
    };
    rng.shuffle(&mut idx);
    idx
}
pub fn stratified(y: &[u8], k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut r = Rng::new(seed);
    let mut assigned = vec![0; y.len()];
    for cls in 0..=1 {
        let mut ids: Vec<_> = (0..y.len()).filter(|&i| y[i] == cls).collect();
        r.shuffle(&mut ids);
        for (j, i) in ids.into_iter().enumerate() {
            assigned[i] = j % k;
        }
    }
    (0..k)
        .map(|f| {
            (
                (0..y.len()).filter(|&i| assigned[i] != f).collect(),
                (0..y.len()).filter(|&i| assigned[i] == f).collect(),
            )
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numpy_oracle() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../tests/rng_oracle.json")).unwrap();
        for c in cases.as_array().unwrap() {
            let seed = c["seed"].as_u64().unwrap();
            let mut r = Rng::new(seed);
            let raw: Vec<_> = (0..8).map(|_| r.raw()).collect();
            assert_eq!(serde_json::json!(raw), c["raw"], "raw seed={seed}");
            let mut a: Vec<_> = (0..17).collect();
            Rng::new(seed).shuffle(&mut a);
            assert_eq!(serde_json::json!(a), c["shuffle"], "shuffle seed={seed}");
            assert_eq!(
                serde_json::json!(Rng::new(seed).choice(13, 17, true)),
                c["replace"],
                "replace seed={seed}"
            );
            assert_eq!(
                serde_json::json!(Rng::new(seed).choice(101, 31, false)),
                c["choice"],
                "choice seed={seed}"
            );
            assert_eq!(
                serde_json::json!(Rng::new(seed).choice(20001, 501, false)),
                c["large_choice"],
                "large choice seed={seed}"
            );
        }
    }
}
