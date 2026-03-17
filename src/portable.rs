use crate::{
    counter_high, counter_low, CVBytes, CVWords, IncrementCounter, BLOCK_LEN, IV, MSG_SCHEDULE,
    OUT_LEN,
};
use arrayref::{array_mut_ref, array_ref};

#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    // Load into locals to avoid repeated bounds checks and help the compiler
    // keep values in registers.
    let mut sa = state[a];
    let mut sb = state[b];
    let mut sc = state[c];
    let mut sd = state[d];

    sa = sa.wrapping_add(sb).wrapping_add(x);
    sd = (sd ^ sa).rotate_right(16);
    sc = sc.wrapping_add(sd);
    sb = (sb ^ sc).rotate_right(12);
    sa = sa.wrapping_add(sb).wrapping_add(y);
    sd = (sd ^ sa).rotate_right(8);
    sc = sc.wrapping_add(sd);
    sb = (sb ^ sc).rotate_right(7);

    state[a] = sa;
    state[b] = sb;
    state[c] = sc;
    state[d] = sd;
}

#[inline(always)]
fn round(state: &mut [u32; 16], msg: &[u32; 16], round: usize) {
    // Select the message schedule based on the round and pre-fetch all
    // message words to avoid repeated indirect indexing.
    let schedule = MSG_SCHEDULE[round];
    let m: [u32; 16] = [
        msg[schedule[0]],
        msg[schedule[1]],
        msg[schedule[2]],
        msg[schedule[3]],
        msg[schedule[4]],
        msg[schedule[5]],
        msg[schedule[6]],
        msg[schedule[7]],
        msg[schedule[8]],
        msg[schedule[9]],
        msg[schedule[10]],
        msg[schedule[11]],
        msg[schedule[12]],
        msg[schedule[13]],
        msg[schedule[14]],
        msg[schedule[15]],
    ];

    // Mix the columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);

    // Mix the diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

#[inline(always)]
fn compress_pre(
    cv: &CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) -> [u32; 16] {
    let block_words = crate::platform::words_from_le_bytes_64(block);

    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter_low(counter),
        counter_high(counter),
        block_len as u32,
        flags as u32,
    ];

    round(&mut state, &block_words, 0);
    round(&mut state, &block_words, 1);
    round(&mut state, &block_words, 2);
    round(&mut state, &block_words, 3);
    round(&mut state, &block_words, 4);
    round(&mut state, &block_words, 5);
    round(&mut state, &block_words, 6);

    state
}

pub fn compress_in_place(
    cv: &mut CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    let state = compress_pre(cv, block, block_len, counter, flags);
    // Use explicit indexing with known bounds so the compiler can elide
    // bounds checks across the entire XOR-fold.
    let (lo, hi) = state.split_at(8);
    for i in 0..8 {
        cv[i] = lo[i] ^ hi[i];
    }
}

pub fn compress_xof(
    cv: &CVWords,
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    counter: u64,
    flags: u8,
) -> [u8; 64] {
    let mut state = compress_pre(cv, block, block_len, counter, flags);
    state[0] ^= state[8];
    state[1] ^= state[9];
    state[2] ^= state[10];
    state[3] ^= state[11];
    state[4] ^= state[12];
    state[5] ^= state[13];
    state[6] ^= state[14];
    state[7] ^= state[15];
    state[8] ^= cv[0];
    state[9] ^= cv[1];
    state[10] ^= cv[2];
    state[11] ^= cv[3];
    state[12] ^= cv[4];
    state[13] ^= cv[5];
    state[14] ^= cv[6];
    state[15] ^= cv[7];
    crate::platform::le_bytes_from_words_64(&state)
}

pub fn hash1<const N: usize>(
    input: &[u8; N],
    key: &CVWords,
    counter: u64,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut CVBytes,
) {
    debug_assert_eq!(N % BLOCK_LEN, 0, "uneven blocks");
    let mut cv = *key;
    // Work over chunks directly; the compiler knows the slice length is a
    // multiple of BLOCK_LEN so it can avoid repeated length checks.
    let chunks = input.chunks_exact(BLOCK_LEN);
    let num_blocks = N / BLOCK_LEN;
    let mut block_flags = flags | flags_start;
    for (i, block) in chunks.enumerate() {
        if i == num_blocks - 1 {
            block_flags |= flags_end;
        }
        compress_in_place(
            &mut cv,
            block.try_into().unwrap(),
            BLOCK_LEN as u8,
            counter,
            block_flags,
        );
        block_flags = flags;
    }
    *out = crate::platform::le_bytes_from_words_32(&cv);
}

pub fn hash_many<const N: usize>(
    inputs: &[&[u8; N]],
    key: &CVWords,
    mut counter: u64,
    increment_counter: IncrementCounter,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: &mut [u8],
) {
    debug_assert!(out.len() >= inputs.len() * OUT_LEN, "out too short");
    let do_increment = increment_counter.yes();
    for (&input, output) in inputs.iter().zip(out.chunks_exact_mut(OUT_LEN)) {
        hash1(
            input,
            key,
            counter,
            flags,
            flags_start,
            flags_end,
            array_mut_ref!(output, 0, OUT_LEN),
        );
        if do_increment {
            counter += 1;
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    // This is basically testing the portable implementation against itself,
    // but it also checks that compress_in_place and compress_xof are
    // consistent. And there are tests against the reference implementation and
    // against hardcoded test vectors elsewhere.
    #[test]
    fn test_compress() {
        crate::test::test_compress_fn(compress_in_place, compress_xof);
    }

    // Ditto.
    #[test]
    fn test_hash_many() {
        crate::test::test_hash_many_fn(hash_many, hash_many);
    }
}