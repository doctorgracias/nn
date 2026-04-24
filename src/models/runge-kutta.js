export function rungeKutta4(f, t0, y0, h, numSteps) {
    let t = t0;
    let y = [...y0];
    const solution = [[t, ...y]];

    for (let i = 0; i < numSteps; i++) {
        const k1 = f(t, y).map(v => v * h);
        const k2 = f(t + 0.5 * h, y.map((v, j) => v + 0.5 * k1[j])).map(v => v * h);
        const k3 = f(t + 0.5 * h, y.map((v, j) => v + 0.5 * k2[j])).map(v => v * h);
        const k4 = f(t + h, y.map((v, j) => v + k3[j])).map(v => v * h);

        y = y.map((v, j) => v + (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6);
        t += h;

        if (y.some(val => isNaN(val) || Math.abs(val) > 1000000)) {
            console.warn("rk exploded at step: ", i);
            break; 
        }

        solution.push([t, ...y]);
    }
    return solution;
}