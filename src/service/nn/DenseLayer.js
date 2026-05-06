import MathOps from "./math/mathOps.js";

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS   = 1e-8;

export class DenseLayer {
    constructor(units, activation = 'tanh') {
        this.units      = units;
        this.activation = activation;
        this.weights    = null;
        this.bias       = null;
        this.input      = null;
        this.z          = null;
        this.output     = null;
        this._mW = null; this._vW = null;
        this._mB = null; this._vB = null;
        this._t  = 0;
    }

    initWeights(fanIn) {
        const std = Math.sqrt(1.0 / fanIn); // Xavier для tanh
        this.weights = Array.from({ length: fanIn }, () =>
            Array.from({ length: this.units }, () => MathOps.randomNormal(0, std))
        );
        this.bias = new Array(this.units).fill(0);
        this._mW = Array.from({ length: fanIn }, () => new Array(this.units).fill(0));
        this._vW = Array.from({ length: fanIn }, () => new Array(this.units).fill(0));
        this._mB = new Array(this.units).fill(0);
        this._vB = new Array(this.units).fill(0);
    }

    forward(input) {
        this.input = input;
        if (!this.weights) this.initWeights(input[0].length);
        this.z      = MathOps.addVec(MathOps.dot(input, this.weights), this.bias);
        this.output = this.activation === 'tanh'
            ? this.z.map(r => r.map(v => Math.tanh(v)))
            : this.z; // linear
        return this.output;
    }

    getDerivative() {
        return this.activation === 'tanh'
            ? this.output.map(r => r.map(v => 1 - v * v))
            : this.z.map(r => r.map(() => 1)); // linear
    }

    backward(gradOutput, lr) {
        const delta       = MathOps.mul(gradOutput, this.getDerivative());
        const gradWeights = MathOps.dot(MathOps.transpose(this.input), delta);
        const gradBias    = delta
            .reduce((acc, row) => acc.map((v, i) => v + row[i]), new Array(this.units).fill(0))
            .map(v => v / delta.length);

        this._t++;
        const bc1 = 1 - Math.pow(BETA1, this._t);
        const bc2 = 1 - Math.pow(BETA2, this._t);

        this.weights = this.weights.map((row, i) =>
            row.map((w, j) => {
                const g = gradWeights[i][j];
                this._mW[i][j] = BETA1 * this._mW[i][j] + (1 - BETA1) * g;
                this._vW[i][j] = BETA2 * this._vW[i][j] + (1 - BETA2) * g * g;
                return w - lr * (this._mW[i][j] / bc1) / (Math.sqrt(this._vW[i][j] / bc2) + EPS);
            })
        );

        this.bias = this.bias.map((b, i) => {
            const g = gradBias[i];
            this._mB[i] = BETA1 * this._mB[i] + (1 - BETA1) * g;
            this._vB[i] = BETA2 * this._vB[i] + (1 - BETA2) * g * g;
            return b - lr * (this._mB[i] / bc1) / (Math.sqrt(this._vB[i] / bc2) + EPS);
        });

        return MathOps.dot(delta, MathOps.transpose(this.weights));
    }
}