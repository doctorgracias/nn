const MathOps = {
    dot: (a, b) => a.map(row => b[0].map((_, i) => row.reduce((acc, _, j) => acc + row[j] * b[j][i], 0))),
    addVec: (matrix, vec) => matrix.map(row => row.map((v, i) => v + vec[i])),
    transpose: (a) => a[0].map((_, i) => a.map(row => row[i])),
    sub: (a, b) => a.map((row, i) => row.map((v, j) => v - b[i][j])),
    mul: (a, b) => a.map((row, i) => row.map((v, j) => v * b[i][j])),

    randomNormal: (mean, std) => {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
};

export default MathOps;