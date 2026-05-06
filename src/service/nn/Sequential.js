import MathOps from "./math/mathOps.js";

export class Sequential {
    constructor(layers) {
        this.layers = layers;
    }

    predict(input) {
        let out = input;
        for (const layer of this.layers) out = layer.forward(out);
        return out;
    }

    trainStep(input, target, lr) {
        const prediction = this.predict(input);
        let grad = MathOps.sub(prediction, target);
        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad, lr);
        }
        const elems = prediction.length * prediction[0].length;
        return prediction
            .map((row, i) => row.reduce((acc, v, j) => acc + (v - target[i][j]) ** 2, 0))
            .reduce((a, b) => a + b, 0) / elems;
    }
}