import MathOps from "./math/mathOps.js";

export class Sequential {
    constructor(layers) {
        this.layers = layers;
        // Adam state: хранится отдельно от слоёв
        this._adamState = null;
        this._adamT = 0;
    }

    predict(input) {
        let output = input;
        for (const layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }

    trainStep(input, target, lr) {
        const prediction = this.predict(input);
        let grad = MathOps.sub(prediction, target);

        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad, lr);
        }

        const totalElems = prediction.length * prediction[0].length;
        return prediction
            .map((row, i) => row.reduce((acc, v, j) => acc + Math.pow(v - target[i][j], 2), 0))
            .reduce((a, b) => a + b, 0) / totalElems;
    }
}