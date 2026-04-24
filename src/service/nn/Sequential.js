import MathOps from "./math/mathOps.js";

export class Sequential {
    constructor(layers) {
        this.layers = layers;
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

    /**
     * @param {number[][]} startNormInput  
     * @param {number[][][]} normTargets
     * @param {number} lr
     * @returns {number} 
     */
    trainStepMulti(startNormInput, normTargets, lr) {
        const K = normTargets.length;
        const predictions = [];
        let currentInput = startNormInput;

        for (let k = 0; k < K; k++) {
            const pred = this.predict(currentInput);  
            const nextInput = [pred[0].map((v, j) => currentInput[0][j] + v)];
            predictions.push({ pred, input: currentInput });
            currentInput = nextInput;
        }
        let totalMse = 0;
        for (let k = 0; k < K; k++) {
            const pred = predictions[k].pred;
            const target = normTargets[k];  
            const elems = pred.length * pred[0].length;
            totalMse += pred
                .map((row, i) => row.reduce((acc, v, j) => acc + Math.pow(v - target[i][j], 2), 0))
                .reduce((a, b) => a + b, 0) / elems;
        }

        for (let k = K - 1; k >= 0; k--) {
            const pred = predictions[k].pred;
            const target = normTargets[k];

            this.predict(predictions[k].input);

            let grad = MathOps.sub(pred, target);
            for (let i = this.layers.length - 1; i >= 0; i--) {
                grad = this.layers[i].backward(grad, lr / K); 
            }
        }

        return totalMse / K;
    }
}