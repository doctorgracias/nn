import MathOps from "./math/mathOps.js";

export class DenseLayer {
    constructor(units, activation = 'relu') {
        this.units = units;
        this.activation = activation;
        this.weights = null;
        this.bias = null;
        this.input = null;
        this.z = null;
        this.output = null;
    }

    initWeights(fanIn) {
        const std = Math.sqrt(1.0 / fanIn);
        this.weights = Array.from({ length: fanIn }, () =>
            Array.from({ length: this.units }, () => MathOps.randomNormal(0, std))
        );
        this.bias = new Array(this.units).fill(0);
    }

    forward(input) {
        this.input = input;
        if (!this.weights) this.initWeights(input[0].length);
        this.z = MathOps.addVec(MathOps.dot(input, this.weights), this.bias);
        this.output = this.applyActivation(this.z);
        return this.output;
    }

    applyActivation(z) {
        if (this.activation === 'relu') return z.map(r => r.map(v => Math.max(0, v)));
        if (this.activation === 'sigmoid') return z.map(r => r.map(v => 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, v))))));
        return z; 
    }

    getDerivative() {
        if (this.activation === 'relu') {
            return this.z.map(r => r.map(v => v > 0 ? 1 : 0));
        }
        if (this.activation === 'sigmoid') {
            return this.output.map(r => r.map(v => v * (1 - v)));
        }
        return this.z.map(r => r.map(() => 1));
    }

    backward(gradOutput, lr) {
        const delta = MathOps.mul(gradOutput, this.getDerivative());

        const gradWeights = MathOps.dot(MathOps.transpose(this.input), delta);

        const gradBias = delta
            .reduce((acc, row) => acc.map((v, i) => v + row[i]), new Array(this.units).fill(0))
            .map(v => v / delta.length);

        this.weights = this.weights.map((row, i) =>
            row.map((v, j) => v - lr * gradWeights[i][j])
        );
        this.bias = this.bias.map((v, i) => v - lr * gradBias[i]);

        return MathOps.dot(delta, MathOps.transpose(this.weights));
    }
}