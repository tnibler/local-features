//! Fast atan2 approximations.
// Taken from: https://mazzo.li/posts/vectorized-atan2.html
// The original license header is reproduced below.
//
// Copyright (c) 2021 Francesco Mazzoli <f@mazzo.li>
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

const float A1 = 0.99997726;
const float A3 = -0.33262347;
const float A5 = 0.19354346;
const float A7 = -0.11643287;
const float A9 = 0.05265332;
const float A11 = -0.0117212;

const float PI = 3.1415927;
const float FRAC_PI_2 = 1.5707964;

float atan2(float x, float y) {
    if (x == 0 && y == 0) {
        return 0;
    }
    const bool swap = abs(x) < abs(y);
    const float a = swap ? (x / y) : (y / x);
    const float asq = a * a;
    const float atan_res = a * (A1 + asq * (A3 + asq * (A5 + asq * (A7 + asq * (A9 + asq * A11)))));
    const float sign_a = -1.0 * uintBitsToFloat(floatBitsToUint(a) & uint(-1));
    const float res = swap ? (FRAC_PI_2 * sign(a) - atan_res) : (atan_res);
    if (sign(x) == -1.0) {
        const float sign_y = 1.0 * uintBitsToFloat((floatBitsToUint(sign(y))) | floatBitsToUint(1.0));
        return PI * sign_y + res;
        // return sign_y;
    } else {
        return res;
    }
}
