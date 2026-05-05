import MathOps from "./math/mathOps.js";

const ADAM_BETA1  = 0.9;
const ADAM_BETA2  = 0.999;
const ADAM_EPS    = 1e-8;

export class DenseLayer {
    constructor(units, activation = 'relu') {
        this.units      = units;
        this.activation = activation;
        this.weights    = null;
        this.bias       = null;
        this.input      = null;
        this.z          = null;
        this.output     = null;

        // Adam моменты
        this._mW = null; this._vW = null;
        this._mB = null; this._vB = null;
        this._t  = 0;
    }

    initWeights(fanIn) {
        // Xavier для tanh: std = sqrt(1/fanIn)
        // He для relu:     std = sqrt(2/fanIn)
        const std = this.activation === 'relu'
            ? Math.sqrt(2.0 / fanIn)
            : Math.sqrt(1.0 / fanIn);

        this.weights = Array.from({ length: fanIn }, () =>
            Array.from({ length: this.units }, () => MathOps.randomNormal(0, std))
        );
        this.bias = new Array(this.units).fill(0);

        // Инициализируем Adam моменты нулями
        this._mW = Array.from({ length: fanIn }, () => new Array(this.units).fill(0));
        this._vW = Array.from({ length: fanIn }, () => new Array(this.units).fill(0));
        this._mB = new Array(this.units).fill(0);
        this._vB = new Array(this.units).fill(0);
    }

    forward(input) {
        this.input = input;
        if (!this.weights) this.initWeights(input[0].length);
        this.z      = MathOps.addVec(MathOps.dot(input, this.weights), this.bias);
        this.output = this.applyActivation(this.z);
        return this.output;
    }

    applyActivation(z) {
        if (this.activation === 'relu')    return z.map(r => r.map(v => Math.max(0, v)));
        if (this.activation === 'sigmoid') return z.map(r => r.map(v => 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, v))))));
        if (this.activation === 'tanh')    return z.map(r => r.map(v => Math.tanh(v)));
        return z;
    }

    getDerivative() {
        if (this.activation === 'relu')    return this.z.map(r => r.map(v => v > 0 ? 1 : 0));
        if (this.activation === 'sigmoid') return this.output.map(r => r.map(v => v * (1 - v)));
        if (this.activation === 'tanh')    return this.output.map(r => r.map(v => 1 - v * v));
        return this.z.map(r => r.map(() => 1));
    }

    backward(gradOutput, lr) {
        const delta      = MathOps.mul(gradOutput, this.getDerivative());
        const gradWeights = MathOps.dot(MathOps.transpose(this.input), delta);
        const gradBias   = delta
            .reduce((acc, row) => acc.map((v, i) => v + row[i]), new Array(this.units).fill(0))
            .map(v => v / delta.length);

        // --- Adam update ---
        this._t++;
        const bc1 = 1 - Math.pow(ADAM_BETA1, this._t);
        const bc2 = 1 - Math.pow(ADAM_BETA2, this._t);

        this.weights = this.weights.map((row, i) =>
            row.map((w, j) => {
                const g = gradWeights[i][j];
                this._mW[i][j] = ADAM_BETA1 * this._mW[i][j] + (1 - ADAM_BETA1) * g;
                this._vW[i][j] = ADAM_BETA2 * this._vW[i][j] + (1 - ADAM_BETA2) * g * g;
                const mHat = this._mW[i][j] / bc1;
                const vHat = this._vW[i][j] / bc2;
                return w - lr * mHat / (Math.sqrt(vHat) + ADAM_EPS);
            })
        );

        this.bias = this.bias.map((b, i) => {
            const g = gradBias[i];
            this._mB[i] = ADAM_BETA1 * this._mB[i] + (1 - ADAM_BETA1) * g;
            this._vB[i] = ADAM_BETA2 * this._vB[i] + (1 - ADAM_BETA2) * g * g;
            const mHat = this._mB[i] / bc1;
            const vHat = this._vB[i] / bc2;
            return b - lr * mHat / (Math.sqrt(vHat) + ADAM_EPS);
        });

        return MathOps.dot(delta, MathOps.transpose(this.weights));
    }
}