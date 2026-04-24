export const systemODE = (t, [x, y, z]) => {
    const dx = -0.2 * y;
    const dy = x + z;
    const dz = x + Math.pow(y, 2) - z;
    return [dx, dy, dz];
};